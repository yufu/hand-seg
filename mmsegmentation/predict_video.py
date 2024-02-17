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


parser = argparse.ArgumentParser(description="")
parser.add_argument("--mode", default='obj1', type=str, help='options: obj1, obj2')
parser.add_argument("--input_video_file", default='../testvideos/testvideo1_short.mp4', type=str)
parser.add_argument("--output_video_file", default='../testvideos/testvideo1_short_result_big_80000s.mp4', type=str)
parser.add_argument("--twohands_config_file", default='./work_dirs/seg_twohands/seg_twohands.py', type=str)
parser.add_argument("--twohands_checkpoint_file", default='./work_dirs/seg_twohands/iter_80000.pth', type=str)
parser.add_argument("--remove_intermediate_images", default=False, type=bool)
args = parser.parse_args()

video_dir = args.input_video_file.split('.mp4')[0]; os.makedirs(video_dir, exist_ok = True)
video_image_dir = os.path.join(video_dir, 'images'); os.makedirs(video_image_dir, exist_ok = True)
video_twohands_dir = os.path.join(video_dir, 'pred_twohands'); os.makedirs(video_twohands_dir, exist_ok = True)
video_cb_dir = os.path.join(video_dir, 'pred_cb'); os.makedirs(video_cb_dir, exist_ok = True)
video_obj1_dir = os.path.join(video_dir, 'pred_obj1'); os.makedirs(video_obj1_dir, exist_ok = True)
video_obj2_dir = os.path.join(video_dir, 'pred_obj2'); os.makedirs(video_obj2_dir, exist_ok = True)


# # extract video frames and save them into a directory
print('Reading and extracting video frames......')
reader = imageio.get_reader(args.input_video_file, 'ffmpeg')
fps = reader.get_meta_data()['fps']
for num, image in enumerate(reader):
    save_img_file = os.path.join(video_image_dir, str(num).zfill(8)+'.jpg')
    imsave(save_img_file, image)

# predict twohands
cmd_pred_twohands = 'python predict_image.py \
                    --config_file %s \
                    --checkpoint_file %s \
                    --img_dir %s \
                    --pred_seg_dir %s' % (args.twohands_config_file, args.twohands_checkpoint_file, video_image_dir, video_twohands_dir)

print('Predicting twohands......')
print(cmd_pred_twohands)
os.system(cmd_pred_twohands)
writer = imageio.get_writer(args.output_video_file, fps = fps)
for img_file in tqdm(sorted(glob.glob(video_image_dir + '/*'))):
    fname = os.path.basename(img_file).split('.')[0]
    twohands_file = os.path.join(video_twohands_dir, fname + '.png')
    img = np.array(Image.open(img_file))
    twohands = np.array(Image.open(twohands_file))
    vis = visualize_twohands(img, twohands)
    writer.append_data(vis)
writer.close()

# remove all folders
if args.remove_intermediate_images:
    os.system('rm -rf ' + video_dir)

