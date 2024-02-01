import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random
import numpy as np
from PIL import Image

def crop_square(frame):
    """
    Crop the largest square from the center of a frame.

    :param frame: A numpy array representing an image.
    :return: Cropped square image.
    """
    height, width, _ = frame.shape
    min_edge = min(height, width)
    top = (height - min_edge) // 2
    left = (width - min_edge) // 2
    return frame[top:top+min_edge, left:left+min_edge]

class VideoDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.segment_length = 8
        self.size = (512,512)
        video_list = 'short_3min.txt'
        video_list = os.path.join(root, video_list)
        self.video_paths = []
        f = open(video_list,encoding='latin1')
        for line in f.readlines():
            line = line.strip().replace("\\","/")
            video_path = os.path.join(root,line)
            self.video_paths.append(video_path)
        f.close()

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure the segment_length does not exceed total_frames
        segment_length = self.segment_length
        if segment_length > total_frames:
            raise ValueError('video %s is too shoot' % video_path)

        # Choose a random start frame for the segment
        start_frame = random.randint(0, total_frames - segment_length)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(segment_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_square(frame)
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),self.size)
            frames.append(frame)

        cap.release()

        # Convert list of frames to a numpy array
        video_array = np.array(frames)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)
        video_tensor = video_tensor / 127.5 - 1.0

        # get id frame 
        id_frame_start = max(0,start_frame - segment_length)
        id_frame_end = min(start_frame + 2*segment_length, total_frames)
        id_frame_ind = random.randint(id_frame_start, id_frame_end)

        # read id frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, id_frame_ind)
        ret, id_frame = cap.read()
        id_frame = crop_square(id_frame)
        id_frame = cv2.resize(cv2.cvtColor(id_frame, cv2.COLOR_BGR2RGB),self.size)
        id_frame = torch.from_numpy(id_frame).permute(2,0,1)
        id_frame = id_frame / 127.5 - 1.0
        
        # print(start_frame, id_frame_ind)

        result = {
            'video':video_tensor,
            'id_frame':id_frame,
        }

        return result

if __name__ == '__main__':
    dataset = VideoDataset('/data3/hzj/coser_dataset/')
    p = dataset[0]
    print(p['video'].shape)

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
    id_frame = (id_frame + 1.0) * 127.5
    id_frame = id_frame.permute(1,2,0).numpy()
    id_frame = id_frame.astype(np.uint8)
    id_frame= Image.fromarray(id_frame)
    id_frame.save('id_frame.jpg')