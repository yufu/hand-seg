#!/bin/bash
python tools/test.py \
work_dirs/seg_twohands_lama_ccda_3cls/seg_twohands_ccda.py \
work_dirs/seg_twohands_lama_ccda_3cls/best_mIoU_iter_86000.pth \
--work-dir work_dirs/seg_twohands_lama_ccda_3cls/ \
--eval mIoU \
--show-dir  work_dirs/seg_twohands_lama_ccda_3cls/best_mIoU_iter_86000_test