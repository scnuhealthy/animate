import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random
import numpy as np
from PIL import Image
from einops import rearrange
from decord import VideoReader

def crop_square(frame):
    """
    Crop the largest square from the center of a frame.

    :param frame: A numpy array representing an image.
    :return: Cropped square image.
    """
    f, height, width, _ = frame.shape
    min_edge = min(height, width)
    top = (height - min_edge) // 2
    left = (width - min_edge) // 2
    return frame[:,top:top+min_edge, left:left+min_edge,:]

class VideoDataset(Dataset):
    def __init__(self, root, sample_n_frames=8, sample_stride=4, size=(512,512),is_image=False):
        self.root = root
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.size = size
        self.is_image = is_image
        video_list = 'short_3min.txt'
        video_list = os.path.join(root, video_list)
        self.video_paths = []
        f = open(video_list,encoding='gbk')
        for line in f.readlines():
            line = line.strip().replace("\\","/")
            video_path = os.path.join(root,line)
            self.video_paths.append(video_path)
        f.close()
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])


    def __getitem__(self, index):
        try:
            video_path = self.video_paths[index]
            video_reader = VideoReader(video_path)
            video_length = len(video_reader)

            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            if self.is_image:
                batch_index = [np.random.choice(batch_index)]
                # batch_index = [random.randint(0, video_length - 1)]
            video_numpy = video_reader.get_batch(batch_index).asnumpy()
            video_numpy = crop_square(video_numpy)

            
            video_tensor = torch.from_numpy(video_numpy).permute(0, 3, 1, 2).contiguous()
            video_tensor = video_tensor / 255
            video_tensor = self.pixel_transforms(video_tensor)

            id_frame_start = max(0,start_idx - clip_length)
            id_frame_end = min(start_idx + 2*clip_length, video_length-1)
            id_frame_idx = [random.randint(id_frame_start, id_frame_end)]

            id_frame_numpy = video_reader.get_batch(id_frame_idx).asnumpy()
            id_frame_numpy = crop_square(id_frame_numpy)
            id_frame_tensor = torch.from_numpy(id_frame_numpy).permute(0, 3, 1, 2).contiguous()
            id_frame_tensor = id_frame_tensor / 255
            id_frame_tensor = self.pixel_transforms(id_frame_tensor)
            id_frame_tensor = id_frame_tensor[0]

            del video_reader

            if self.is_image:
                video_tensor = video_tensor[0]
            # print(id_frame_start, id_frame_end, id_frame_idx, video_length)
            result = {
                'video':video_tensor,
                'id_frame':id_frame_tensor,
            }
        except:
            return self.__getitem__(index+1)
        return result

    def __len__(self):
        return len(self.video_paths)

if __name__ == '__main__':
    dataset = VideoDataset('/data3/hzj/coser_dataset/',is_image=False)
    p = dataset[0]
    print(p['video'].shape)

    # video = p['video']
    # video = (video + 1.0) * 127.5
    # image = video
    # image = image.permute(1,2,0).numpy()
    # image = image.astype(np.uint8)
    # image= Image.fromarray(image)
    # image.save('image.jpg')

    # id_frame = p['id_frame']
    # id_frame = (id_frame + 1.0) * 127.5
    # id_frame = id_frame.permute(1,2,0).numpy()
    # id_frame = id_frame.astype(np.uint8)
    # id_frame= Image.fromarray(id_frame)
    # id_frame.save('id_frame.jpg')

    video = p['video']
    video = (video + 1.0) * 127.5
    image = video[0]
    image = image.permute(1,2,0).numpy()
    image = image.astype(np.uint8)
    image= Image.fromarray(image)
    image.save('image.jpg')

    image = video[5]
    image = image.permute(1,2,0).numpy()
    image = image.astype(np.uint8)
    image= Image.fromarray(image)
    image.save('image5.jpg')

    id_frame = p['id_frame']
    print(id_frame.shape)
    id_frame = (id_frame + 1.0) * 127.5
    id_frame = id_frame.permute(1,2,0).numpy()
    id_frame = id_frame.astype(np.uint8)
    id_frame= Image.fromarray(id_frame)
    id_frame.save('id_frame.jpg')