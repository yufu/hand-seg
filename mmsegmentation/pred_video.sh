python predict_video_withimage.py \
       --twohands_config_file ./work_dirs/seg_twohands_lama_ccda_3clsPaste_2labelPred_3datasets/seg_twohands_ccda.py \
       --twohands_checkpoint_file ./work_dirs/seg_twohands_lama_ccda_3clsPaste_2labelPred_3datasets/best_mIoU_iter_150000.pth \
       --input_video_folder test_videos \
       --output_video_folder ./work_dirs/seg_twohands_lama_ccda_3clsPaste_2labelPred_3datasets/15000 \
       --save_image True