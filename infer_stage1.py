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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, DDIMScheduler
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from models.unet import UNet3DConditionModel
from models.condition_encoder import FrozenOpenCLIPImageEmbedderV2
from omegaconf import OmegaConf
from pipelines.pipeline_stage1 import AnimateStage1Pipeline
from models.hack_poseguider import Hack_PoseGuider as PoseGuider

from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention_fp16 import ReferenceNetAttention
from models.ReferenceEncoder import ReferenceEncoder
from data.dataset import TikTok, collate_fn, UBC_Fashion
from data.talk_dataset import Talk_Dataset
from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel

config = OmegaConf.load('config/infer_stage1.yaml')

# seed 
seed = config.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# dataset
infer_data_config = config.infer_data
infer_dataset = Talk_Dataset(**infer_data_config,is_image=config.image_finetune)
# infer_dataset = TikTok(**infer_data_config,is_image=config.image_finetune)
# infer_dataset = UBC_Fashion(**infer_data_config,is_image=config.image_finetune)

test_dataloader = torch.utils.data.DataLoader(
    infer_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=config.batch_size,
    num_workers=config.dataloader_num_workers,
)

unet = UNet2DConditionModel.from_pretrained(
config.unet_path, subfolder="unet",torch_dtype=torch.float16
).to("cuda")
# unet = UNet2DConditionModel.from_pretrained(
# config.unet_path, subfolder=None,torch_dtype=torch.float16
# ).to("cuda")

vae= AutoencoderKL.from_pretrained(
    config.vae_path, subfolder="vae",torch_dtype=torch.float16
).to("cuda")

referencenet = ReferenceNet.from_pretrained(
config.pretrained_referencenet_path, subfolder="referencenet",torch_dtype=torch.float16
).to("cuda")
# referencenet = ReferenceNet.load_referencenet(pretrained_model_path=config.pretrained_referencenet_path).to("cuda", dtype=torch.float16)

pose_guider = PoseGuider.from_pretrained(pretrained_model_path=config.pretrained_poseguider_path).to("cuda", dtype=torch.float16)
pose_guider.eval()
scheduler = DDIMScheduler.from_pretrained(config.model_path, subfolder='scheduler')

pipe = AnimateStage1Pipeline(pose_guider=pose_guider, referencenet=referencenet, vae=vae, unet=unet, scheduler=scheduler)
pipe.enable_xformers_memory_efficient_attention()
# pipe._execution_device = torch.device("cuda")
# pipe.to("cuda")

clip_image_encoder = ReferenceEncoder(model_path=config.clip_model_path).to(device='cuda',dtype=torch.float16)

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
guidance_scale = config.guidance_scale
weight_dtype = torch.float16

# check vae reconstruction
# image_idx = 0
# for i, batch in enumerate(test_dataloader):
#     video = batch['pixel_values'].to(device='cuda', dtype=torch.float16)
#     out = video[0].cpu() /2 +0.5
#     out = out.detach().permute(1,2,0).numpy()
#     out = (out * 255).astype(np.uint8)
#     out = Image.fromarray(out)
#     out.save('%d_test_ori.png' % i)

#     latents = vae.encode(video)
#     latents = latents.latent_dist.sample()

#     reconstruct_video = vae.decode(latents).sample

#     reconstruct_video = reconstruct_video.clamp(-1, 1)
#     out = reconstruct_video[0].cpu() /2 +0.5
#     out = out.detach().permute(1,2,0).numpy()
#     out = (out * 255).astype(np.uint8)
#     out = Image.fromarray(out)
#     out.save('%d_test2.png' % i)


image_idx = 0
for i, batch in enumerate(test_dataloader):

    pixel_values = batch["pixel_values"]
    pixel_values_pose = batch["pixel_values_pose"].to(device='cuda')
    clip_ref_image = batch["clip_ref_image"].to(device='cuda')
    pixel_values_ref_img = batch["pixel_values_ref_img"].to(device='cuda')

    dino_fea = clip_image_encoder(clip_ref_image.to(weight_dtype))
    # dino_fea = dino_fea.unsqueeze(1)
    print(dino_fea.shape) # [bs,1,768]
    edited_images = pipe(
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale, 
        source_image=pixel_values_ref_img.to(weight_dtype),
        pose_image=pixel_values_pose.to(weight_dtype),
        generator=generator,
        dino_fea = dino_fea,
    ).images

    for idx, edited_image in enumerate(edited_images):
        edited_image = torch.tensor(np.array(edited_image)).permute(2,0,1) / 255.0
        grid = make_image_grid([(pixel_values[idx].cpu() / 2 + 0.5),edited_image.cpu(), (pixel_values_pose[idx].cpu() / 2 + 0.5), (pixel_values_ref_img[idx].cpu() / 2 + 0.5)], nrow=2)
        save_image(grid, os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
        # save_image(grid, os.path.join(out_dir, name2))
        image_idx +=1


