##### 一些笔记 #####

关于 do_classifier_free_guidance得实现，首先基于一个思路，讲latent (b,c,h,w)复制为(2b,c,h,w),前面一半是uncond,后面一半是正常得cond
* reference_control_reader: uncond得latent沿用self-attention（不加入ref特征）,然后cond得latent用animateanyone得实现，即合拼referencenet的特征一起做self-attn
* ref_image_latents，latent_pose 复制
* noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)\
   noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)\
   这里chunk是将(2b,c,h,w)拆为两个(b,c,h,w)，分别代表uncond和cond
