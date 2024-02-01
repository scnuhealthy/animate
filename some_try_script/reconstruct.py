import PIL
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
import time
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, DDIMScheduler
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from models.unet import UNet3DConditionModel
from models.condition_encoder import FrozenOpenCLIPImageEmbedderV2
from omegaconf import OmegaConf
from pipelines.pipeline_image import ImagePipeline
from VideoDataset import VideoDataset

config = OmegaConf.load('infer.yaml')

# seed 
seed = config.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def collate_fn(examples):
    video = torch.stack([example["video"] for example in examples])
    id_frame = torch.stack([example["id_frame"] for example in examples])
    return {
        "video":video,
        "id_frame":id_frame,
    }

# dataset
infer_data_config = config.infer_data
infer_dataset = VideoDataset(**infer_data_config)

test_dataloader = torch.utils.data.DataLoader(
    infer_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=1,
    num_workers=2,
)


if config.vae_path is not None:
    vae= AutoencoderKL.from_pretrained(
       config.vae_path, subfolder="vae",torch_dtype=torch.float16
    ).to("cuda")

generator = torch.Generator("cuda").manual_seed(seed)

# infer
out_dir = config.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

weight_dtype = torch.float16

image_idx = 0
for i, batch in enumerate(test_dataloader):
    video = batch['video'].to(device='cuda', dtype=torch.float16)
    out = video[0].cpu() /2 +0.5
    out = out.detach().permute(1,2,0).numpy()
    out = (out * 255).astype(np.uint8)
    out = Image.fromarray(out)
    out.save('%d_test_ori.png' % i)

    latents = vae.encode(video)
    latents = latents.latent_dist.sample()

    reconstruct_video = vae.decode(latents).sample

    reconstruct_video = reconstruct_video.clamp(-1, 1)
    out = reconstruct_video[0].cpu() /2 +0.5
    out = out.detach().permute(1,2,0).numpy()
    out = (out * 255).astype(np.uint8)
    out = Image.fromarray(out)
    out.save('%d_test.png' % i)

