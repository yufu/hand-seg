import os 
import glob
import pdb
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser(description="resize image using bicubic upsampling")
parser.add_argument("--img_dir", default='/impacs/yuf5/EgoHOS/data/lama_512_feature', type=str)

clean_type='mask.npy'
args = parser.parse_args()



for file in tqdm(glob.glob(args.img_dir + '/*')):
    ext = file.split('.')[-1]
    val_list = (file.split('/')[-1].split('.')[0] + '.' + file.split('/')[-1].split('.')[1]).split('_')

    # pdb.set_trace()
 
    if clean_type in val_list:
            
        fname = val_list[0]
        for i in range(1,len(val_list)-1):
            fname = fname + '_' + val_list[i]
        
        dst_file = os.path.join(args.img_dir, fname + '.' + ext)

        os.rename(file, dst_file)

        # pdb.set_trace()

