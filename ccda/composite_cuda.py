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

parser = argparse.ArgumentParser(description="composite")
parser.add_argument("--aug_numbers", default=3, type=int)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--random_aug", default=False, type=bool)
parser.add_argument("--composite_hr", default=True, type=bool)
parser.add_argument("--hr_img_dir", default='/impacs/yuf5/EgoHOS/data/train/image', type=str)
parser.add_argument("--hr_lbl_dir", default='/impacs/yuf5/EgoHOS/data/train/label', type=str)
parser.add_argument("--img_dir", default='/impacs/yuf5/EgoHOS/data/train/image', type=str)
parser.add_argument("--lbl_dir", default='/impacs/yuf5/EgoHOS/data/train/label', type=str)
parser.add_argument("--lama_dir", default='/impacs/yuf5/EgoHOS/data/background_lama', type=str)
parser.add_argument("--lama_feat_dir", default='/impacs/yuf5/EgoHOS/data/lama_512_feature', type=str)
parser.add_argument("--aug_img_dir", default='/impacs/yuf5/EgoHOS/data/train/image_lama_ccda', type=str)
parser.add_argument("--aug_lbl_dir", default='/impacs/yuf5/EgoHOS/data/train/label_lama_ccda', type=str)
args = parser.parse_args()

os.system('rm -rf ' + args.aug_img_dir); os.makedirs(args.aug_img_dir, exist_ok = True)
os.system('rm -rf ' + args.aug_lbl_dir); os.makedirs(args.aug_lbl_dir, exist_ok = True)

# Load combine features into a list
print('Loading features')
fname_list = []
feature_list = []
features = torch.empty((0)).to('cuda')
num = 0
for file in glob.glob(args.lama_feat_dir + '/*'):
    fname = os.path.basename(file).split('.')[0]
    feature = np.load(file)
    num += 1
    if num%100 == 0:
        print(num)
    # print(feature.shape)
    features = torch.cat((features, torch.tensor(feature).unsqueeze(0).to('cuda')))
    fname_list.append(fname)

print("composite images and generate labels")
# composite images and generate labels
for file in tqdm(glob.glob(args.lama_feat_dir + '/*')): 

    query_fname = os.path.basename(file).split('.')[0]
    query_feature = torch.tensor(np.load(file)).to('cuda')

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity = cos(features, query_feature.unsqueeze(0))

    sorted_index = torch.argsort(cosine_similarity, descending=True)

    if args.random_aug:
        top_k_sorted_index_list = sorted_index
    else:
        top_k_sorted_index_list = sorted_index[1:args.top_k+1]
    
    for aug_idx in range(args.aug_numbers):
        
        # size_match = False
        # while not size_match:
        sample_idx = random.randint(0, len(top_k_sorted_index_list)-1)
        index = top_k_sorted_index_list[sample_idx]
        select_fname = fname_list[index]
        #
        #     query_img = Image.open(os.path.join(args.img_dir, query_fname + '.jpg'))
        #     query_img = np.array(query_img)
        #     query_lbl = np.array(Image.open(os.path.join(args.lbl_dir, query_fname + '.png')))
        #     query_msk = np.zeros((query_lbl.shape)); query_msk[query_lbl>0] = 1
        #     query_msk = np.repeat(np.expand_dims(query_msk, 2), 3, 2)
        #     select_img = np.array(Image.open(os.path.join(args.lama_dir, select_fname + '.png')))
        #     if query_img.shape == select_img.shape:
        #         size_match = True
        
        if args.composite_hr:
            hr_query_img = np.array(Image.open(os.path.join(args.hr_img_dir, query_fname + '.jpg')))
            hr_query_lbl = np.array(Image.open(os.path.join(args.hr_lbl_dir, query_fname + '.png')))
            hr_query_msk = np.zeros((hr_query_lbl.shape)); hr_query_msk[hr_query_lbl>0] = 1
            hr_query_msk = np.repeat(np.expand_dims(hr_query_msk, 2), 3, 2)
            hr_select_img = np.array(Image.open(os.path.join(args.lama_dir, select_fname + '.png')).resize((hr_query_img.shape[1], hr_query_img.shape[0])))
            new_img = hr_query_img * hr_query_msk + hr_select_img * (1 - hr_query_msk)
        else:
            new_img = query_img * query_msk + select_img * (1 - query_msk)
        new_img = new_img.astype(np.uint8)
        imsave(os.path.join(args.aug_img_dir, query_fname + '_' + str(aug_idx) + '.jpg'), new_img)
        src_lbl_file = os.path.join(args.lbl_dir, query_fname + '.png')
        dst_lbl_file = os.path.join(args.aug_lbl_dir, query_fname + '_' + str(aug_idx) + '.png')
        copyfile(src_lbl_file, dst_lbl_file)
    
    src_ori_img_file = os.path.join(args.img_dir, query_fname + '.jpg')
    dst_ori_img_file = os.path.join(args.aug_img_dir, query_fname + '.jpg')
    copyfile(src_ori_img_file, dst_ori_img_file)
    
    src_ori_lbl_file = os.path.join(args.lbl_dir, query_fname + '.png')
    dst_ori_lbl_file = os.path.join(args.aug_lbl_dir, query_fname + '.png')
    copyfile(src_ori_lbl_file, dst_ori_lbl_file)



    
    # pdb.set_trace()