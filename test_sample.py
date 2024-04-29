import os
 
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

clear_output(wait=False)



def get_attention_scores_neuron(self, query, key, attn_mask):    
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
  
    return attention_probs
 

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled
 

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple
    
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
 
    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)
    

class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(text_embeds=out_tuple[0], last_hidden_state=out_tuple[1], hidden_states=out_tuple[2])
    
class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        out_tuple = self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
        return out_tuple
    

COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024'

# Model ID for SD XL version pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# --- Compile Text Encoders and save ---

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)


# Apply wrappers to make text encoders traceable
traceable_text_encoder = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
traceable_text_encoder_2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))

del pipe

text_input_ids_1 = torch.tensor([[49406,   736,  1615, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])


text_input_ids_2 = torch.tensor([[49406,   736,  1615, 49407,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]])


# Text Encoder 1
neuron_text_encoder = torch_neuronx.trace(
    traceable_text_encoder,
    text_input_ids_1,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)

text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch.jit.save(neuron_text_encoder, text_encoder_filename)


# Text Encoder 2
neuron_text_encoder_2 = torch_neuronx.trace(
    traceable_text_encoder_2,
    text_input_ids_2,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2'),
)

text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
torch.jit.save(neuron_text_encoder_2, text_encoder_2_filename)

# --- Compile VAE decoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
decoder = copy.deepcopy(pipe.vae.decoder)
del pipe

# # Compile vae decoder
decoder_in = torch.randn([1, 4, 128, 128])
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder


# --- Compile UNet and save ---

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)

# Replace original cross-attention module with custom cross-attention module for better performance
Attention.get_attention_scores = get_attention_scores_neuron

# Apply double wrapper to deal with custom return type
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet = copy.deepcopy(pipe.unet.unetwrap)
del pipe

# Compile unet - FP32
sample_1b = torch.randn([1, 4, 128, 128])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 2048])
added_cond_kwargs_1b = {"text_embeds": torch.randn([1, 1280]),
                        "time_ids": torch.randn([1, 6])}
example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs_1b["text_embeds"], added_cond_kwargs_1b["time_ids"],)

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# save compiled unet
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
torch.jit.save(unet_neuron, unet_filename)

# delete unused objects
del unet


# --- Compile VAE post_quant_conv and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

# Compile vae post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128])
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

# Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv

# --- Load all compiled models and run pipeline ---
COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
pipe.text_encoder = TextEncoderOutputWrapper(torch.jit.load(text_encoder_filename), pipe.text_encoder)
pipe.text_encoder_2 = TextEncoderOutputWrapper(torch.jit.load(text_encoder_2_filename), pipe.text_encoder_2)

# Run pipeline
prompt = ["a photo of an astronaut riding a horse on mars",
          "sonic on the moon",
          "elvis playing guitar while eating a hotdog",
          "saved by the bell",
          "engineers eating lunch at the opera",
          "panda eating bamboo on a plane",
          "A digital illustration of a steampunk flying machine in the sky with cogs and mechanisms, 4k, detailed, trending in artstation, fantasy vivid colors",
          "kids playing soccer at the FIFA World Cup"
         ]

# First do a warmup run so all the asynchronous loads can finish
image_warmup = pipe(prompt[0]).images[0]

plt.title("Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")

total_time = 0
for x in prompt:
    start_time = time.time()
    image = pipe(x).images[0]
    total_time = total_time + (time.time()-start_time)
    image.save("image.png")
    image = mpimg.imread("image.png")

print("Average time: ", np.round((total_time/len(prompt)), 2), "seconds")