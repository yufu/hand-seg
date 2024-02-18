import cv2
import os

# 视频文件路径
video_path = './test_videos_result_fy_56000/full_2.mp4'
# 保存帧的目录
save_dir = './test_videos_result_56000/full_2'

# 检查保存目录是否存在，如果不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 使用OpenCV打开视频
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        break

    # 构建保存帧的文件名
    save_path = os.path.join(save_dir, f'frame_{frame_count:04d}.jpg')

    # 保存帧到文件
    cv2.imwrite(save_path, frame)

    frame_count += 1

# 释放视频捕获对象
cap.release()

print(f'所有帧已保存到 {save_dir}，共 {frame_count} 帧。')
