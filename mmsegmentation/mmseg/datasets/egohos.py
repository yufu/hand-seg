# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class EgoHOSDataset(CustomDataset):
    """EgoHOS dataset.

    Args:
        split (str): Split txt file for EgoHOS.
    """
    
    # CLASSES = ('background', 'aeroplane')

    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128]]

    CLASSES = ('background', 'Left_Hand', 'Right_Hand')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]

    def __init__(self, **kwargs):
        super(EgoHOSDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)



