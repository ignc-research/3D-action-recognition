## 3D-action-recognition

This reposiory contains all the necessary files and instructions required to launch the human action recogniton software. The docker container can be deployed on any Nvidia Jetson board using the Dockerfile. The ZED2 camera is used and the  application is integrated into the ROS ecosystem.

# Setup

First download these two pre-trained models from mmaction2 and copy them to zed_catkin_ws/src/mmdetection_ros/checkpoints

- slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217-ae225e97.pth

- slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth

# Docker 

1. Build: sudo docker build -t ros-zed2-mm-action-recognition:1.0.0 . 

2. Run: sudo docker run -it --runtime nvidia --gpus all --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix:rw ros-zed2-mm-action-recognition:1.0.0 /bin/bash

# Run the software within the docker container

- roslaunch tracking_humans_and_robots_3d zed_people_robot_tracking.launch

# Tips:

- If docker access problems submerge: Run 'xhost +'

- When pulling the repository, use 'git lfs pull'

- If mmaction2 causes an error because of a missing README file then download this file: docs_zh_CN/README.md

- It is easier, better and faster to build the docker container once and work on a commit (docker commit) while developping (This avoids waiting for the ZED2 camera to download and optimize the human detection model for the hardware whenever it is necessary to rebuild the container since the ZED2 camera often freezes/crashes): 

- It is necessary to modify the line 86 in nano /usr/local/lib/python3.7/dist-packages
/mmdet/core/bbox/transforms.py: bbox_list -> bbox_list[0]. Otherwise the action recognition inference will deliver an error. 
