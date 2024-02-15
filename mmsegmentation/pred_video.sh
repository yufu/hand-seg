python predict_video.py \
       --mode obj1 \
       --input_video_file ../testvideos/testvideo1_short.mp4 \
       --output_video_file ../testvideos/testvideo1_short_result.mp4 \
       --remove_intermediate_images True \
       --twohands_checkpoint_file ./work_dirs/seg_twohands/best_mIoU_iter_60000.pth
