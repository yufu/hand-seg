from PIL import Image
import numpy as np 
from skimage.io import imsave
import glob
import pdb
import os 
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import random
from shutil import copyfile
import argparse
import torch
import random

parser = argparse.ArgumentParser(description="composite")
parser.add_argument("--aug_numbers", default=3, type=int)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--random_aug", default=True, type=bool)
parser.add_argument("--composite_hr", default=True, type=bool)
parser.add_argument("--img_dir", default='/impacs/yuf5/EgoHOS/data/train/image', type=str)
parser.add_argument("--lbl_dir", default='/impacs/yuf5/EgoHOS/data/train/label3', type=str)
parser.add_argument("--lama_dir", default='/impacs/yuf5/EgoHOS/data/background_all', type=str)
parser.add_argument("--lama_feat_dir", default='/impacs/yuf5/EgoHOS/data/lama_512_feature_all', type=str)
parser.add_argument("--aug_img_dir", default='/impacs/yuf5/EgoHOS/data/train/image_ccda_all_3cls', type=str)
parser.add_argument("--aug_lbl_dir", default='/impacs/yuf5/EgoHOS/data/train/label_ccda_all_3cls', type=str)
args = parser.parse_args()

os.system('rm -rf ' + args.aug_img_dir); os.makedirs(args.aug_img_dir, exist_ok = True)
os.system('rm -rf ' + args.aug_lbl_dir); os.makedirs(args.aug_lbl_dir, exist_ok = True)

query_files = []
for root, dirs, paths in os.walk(args.img_dir):
    for path in paths:
        if  path.split(".")[1] == 'jpg':
            query_files.append(path.split('.')[0])
print(f'Origin image {len(query_files)} loaded')

bg = []
for root, dirs, paths in os.walk(args.lama_dir):
    for path in paths:
        if  path.split(".")[1] == 'jpg':
            bg.append(os.path.join(root, path))
print(f'Background images:  {len(bg)}')



# composite images and generate labels
for file in tqdm(query_files):
    indexes = []
    for i in range(args.aug_numbers):
        random_integer = random.randint(0, len(bg) - 1)
        indexes.append(random_integer)

    for aug_idx in range(args.aug_numbers):

        size_match = False
        while not size_match:
            select_fname = bg[indexes[aug_idx]]
            query_img = Image.open(os.path.join(args.img_dir, file + '.jpg'))
            shape = query_img.size
            query_img = np.array(query_img)
            query_lbl = np.array(Image.open(os.path.join(args.lbl_dir, file + '.png')))
            query_msk = np.zeros((query_lbl.shape)); query_msk[query_lbl>0] = 1
            query_msk = np.repeat(np.expand_dims(query_msk, 2), 3, 2)
            select_img = np.array(Image.open(select_fname).resize(shape))
            if query_img.shape == select_img.shape:
                size_match = True

            new_img = query_img * query_msk + select_img * (1 - query_msk)
            new_img = new_img.astype(np.uint8)

        
        imsave(os.path.join(args.aug_img_dir, file + '_' + str(aug_idx) + '.jpg'), new_img)
        src_lbl_file = os.path.join(args.lbl_dir, file + '.png')
        dst_lbl_file = os.path.join(args.aug_lbl_dir, file + '_' + str(aug_idx) + '.png')
        copyfile(src_lbl_file, dst_lbl_file)
    
    src_ori_img_file = os.path.join(args.img_dir, file + '.jpg')
    dst_ori_img_file = os.path.join(args.aug_img_dir, file + '.jpg')
    copyfile(src_ori_img_file, dst_ori_img_file)
    
    src_ori_lbl_file = os.path.join(args.lbl_dir, file + '.png')
    dst_ori_lbl_file = os.path.join(args.aug_lbl_dir, file + '.png')
    copyfile(src_ori_lbl_file, dst_ori_lbl_file)



    
    # pdb.set_trace()