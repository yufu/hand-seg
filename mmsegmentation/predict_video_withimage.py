from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
from skimage.io import imsave
import pdb
import imageio
from visualize import visualize_twohands
import cv2
import os

# test file
folder = 'seg_twohands_lama_ccda_3clsPaste_2labelPred'
iter_n = '92000'
best = True

if best == True:
    prefix = 'best_mIoU_iter_'
else:
    prefix='iter_'


parser = argparse.ArgumentParser(description="")
parser.add_argument("--mode", default='obj1', type=str, help='options: obj1, obj2')
parser.add_argument("--twohands_config_file", default=f'./work_dirs/{folder}/seg_twohands_ccda.py', type=str)
parser.add_argument("--twohands_checkpoint_file", default=f'./work_dirs/{folder}/{prefix}{iter_n}.pth', type=str)
parser.add_argument("--remove_intermediate_images", default=False, type=bool)
parser.add_argument("--input_video_folder",  default='test_videos', type=str)
parser.add_argument("--output_video_folder",  default=f'./work_dirs/{folder}/{iter_n}', type=str)
parser.add_argument("--save_image",  default=False, type=bool)

args = parser.parse_args()



def segment_single_video(input_file, output_file, save_image=False):
    print('Reading and extracting video frames......')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    ret, frame = cap.read()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_count += 1
        if frame_count % 100 == 0:
            print(frame_count)
        image = np.array(frame)
        twohands_jpg = inference_segmentor(model,image )[0]
        vis = visualize_twohands(image, twohands_jpg)
        out.write(vis)
        if save_image == True:
            save_folder = os.path.join(args.output_video_folder , input_file.split('/')[1].split('.')[0] )
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            save_path = os.path.join(save_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(save_path, vis)
    cap.release()
    out.release()

# build the model from a config file and a checkpoint file
model = init_segmentor(args.twohands_config_file, args.twohands_checkpoint_file, device='cuda:0')
out_dir = args.output_video_folder
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for root, dirs, files in os.walk(args.input_video_folder):
    for file in files:
        file_path = os.path.join(root, file)
        write_path = os.path.join(out_dir, file)
        print(file_path)
        segment_single_video(file_path, write_path, save_image=args.save_image)
