_base_ = './deeplabv3_r50-d8_4xb4-160k_ade20k-512x512.py'
model = dict(pretrained='./pretrained/', backbone=dict(depth=101))