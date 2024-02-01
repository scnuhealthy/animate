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
# from pipelines.pipeline_image import ImagePipeline
from pipelines.pipeline_image_controlnet import ImageControlPipeline
from models.controlnet import ControlNetModel
from VideoDataset import VideoDataset
from models.unet_2d_controlnet import UNet2DControlNet
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
    batch_size=config.batch_size,
    num_workers=config.dataloader_num_workers,
)

unet = UNet2DControlNet.from_pretrained(
config.unet_path, subfolder="unet",torch_dtype=torch.float16
).to("cuda")

vae= AutoencoderKL.from_pretrained(
    config.vae_path, subfolder="vae",torch_dtype=torch.float16
).to("cuda")

controlnet = ControlNetModel.from_pretrained(config.controlnet_path, subfolder="controlnet", torch_dtype=torch.float16).to("cuda")
scheduler = DDIMScheduler.from_pretrained(config.model_path, subfolder='scheduler')

pipe = ImageControlPipeline(controlnet=controlnet, vae=vae, unet=unet, scheduler=scheduler)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")
# pipe._execution_device = torch.device("cuda")

image_embedder = FrozenOpenCLIPImageEmbedderV2(freeze=True, model_path=config.pretrained_clip_path).to(device='cuda',dtype=torch.float16)

pipe.scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    )
generator = torch.Generator("cuda").manual_seed(seed)

# infer
out_dir = config.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

num_inference_steps = 50
image_guidance_scale = 1
masked_image_guidance_scale = 1
weight_dtype = torch.float16

image_idx = 0
for i, batch in enumerate(test_dataloader):
    video = batch['video'].to(device='cuda')
    video_length = video.shape[1]
    id_frame = batch["id_frame"].to(device='cuda', dtype=weight_dtype)
    print(video.shape, id_frame.shape)
    dino_fea = image_embedder(id_frame)
    print(dino_fea.shape)
    edited_images = pipe(
        num_inference_steps=num_inference_steps, 
        image_guidance_scale=image_guidance_scale, 
        masked_image_guidance_scale=masked_image_guidance_scale,
        generator=generator,
        control_image = id_frame,
        dino_fea = dino_fea,
    ).images

    for idx, edited_image in enumerate(edited_images):
        edited_image = torch.tensor(np.array(edited_image)).permute(2,0,1) / 255.0
        grid = make_image_grid([(id_frame[idx].cpu() / 2 + 0.5),edited_image.cpu()], nrow=1)
        save_image(grid, os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
        # save_image(grid, os.path.join(out_dir, name2))
        image_idx +=1


