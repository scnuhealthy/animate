import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dwpose_utils import DWposeDetector
from decord import VideoReader
from decord import cpu

def process_video(dwprocessor, video_path, output_video_path_frame, output_video_path_pose, detect_resolution):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()

    first_frame = vr[0].asnumpy()
    height, width, _ = first_frame.shape
    size = (width, height)
    
    # 创建视频写入器，使用原视频的帧率
    video_writer_frame = cv2.VideoWriter(output_video_path_frame, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    video_writer_pose = cv2.VideoWriter(output_video_path_pose, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    frame_skip = 8
    for idx in tqdm(range(0, len(vr), frame_skip), desc=f"Processing {os.path.basename(video_path)}"):
        frame = vr[idx].asnumpy()
        frame = frame[...,::-1]
        detected_pose = process(dwprocessor, frame, detect_resolution)
        video_writer_frame.write(frame)
        video_writer_pose.write(detected_pose)

    video_writer_frame.release()
    video_writer_pose.release()

def process(dwprocessor, input_image, detect_resolution):
    if not isinstance(dwprocessor, DWposeDetector):
        dwprocessor = DWposeDetector()

    with torch.no_grad():
        detected_map = dwprocessor(input_image)
    return detected_map

dwprocessor = DWposeDetector()
# dataset_folder = '../../UBC_dataset'
dataset_folder = '/root/autodl-tmp/Talk_dataset/'
sub_folders = ['test']
detect_resolution = 768

for sub_folder in sub_folders:
    path = os.path.join(dataset_folder, sub_folder)
    new_sub_folder = 'deal_' + sub_folder
    output_folder_frame = os.path.join(dataset_folder, new_sub_folder )
    output_folder_pose = os.path.join(dataset_folder, new_sub_folder + '_dwpose')
    if not os.path.exists(output_folder_frame):
        os.makedirs(output_folder_frame)
    if not os.path.exists(output_folder_pose):
        os.makedirs(output_folder_pose)

    for video_name in tqdm(os.listdir(path), desc=f"Processing {sub_folder}"):
        video_path = os.path.join(path, video_name)
        output_video_path_frame = os.path.join(output_folder_frame, video_name.split('.')[0] + '.mp4')
        output_video_path_pose = os.path.join(output_folder_pose, video_name.split('.')[0] + '.mp4')
        process_video(dwprocessor, video_path, output_video_path_frame, output_video_path_pose, detect_resolution)
