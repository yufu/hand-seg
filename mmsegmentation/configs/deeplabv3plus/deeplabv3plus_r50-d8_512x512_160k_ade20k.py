_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/egohos_twohands.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
