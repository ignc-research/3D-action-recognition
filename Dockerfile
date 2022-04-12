
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.5.0
FROM ${BASE_IMAGE}

ARG ROS_PKG=desktop-full
ENV ROS_DISTRO=melodic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=2
ENV OPENCV_VERSION=4.5.3

ENV OPENBLAS_CORETYPE=ARMV8
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential curl wget gnupg2 \
    lsb-release ca-certificates software-properties-common \ 
    && add-apt-repository universe && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# **********************
# Installing python 3.7*
# **********************
ARG PYTHON=python3.7
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON} \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && ${PYTHON} get-pip.py \ 
    && rm get-pip.py 


# ************************************************
# Setting python version to 3.7 as system python *
# and upgrading pip and setuptools               *
# ************************************************
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \ 
    && pip3 --no-cache-dir install --upgrade pip setuptools \
    && ln -s $(which ${PYTHON}) /usr/local/bin/python \
    && apt-get install -y nano ${PYTHON}-dev \
    && python3 -m pip install cython \ 
    && python3 -m pip install numpy  \
    && pip3 install scikit-build setuptools wheel matplotlib cython pyopengl funcy\ 
    && apt-get clean all \ 
    && rm -rf /var/lib/apt/lists/*


# ***************************************
# install zed sdk and setup the ZED SDK *
# ***************************************
COPY ./ZED_SDK_Tegra_JP45_v3.5.5.run . 
RUN apt-get update -y && apt-get install --no-install-recommends \
        lsb-release wget less udev sudo apt-transport-https \
        libqt5xml5 libxmu-dev libxi-dev build-essential cmake -y \
    && chmod +x ZED_SDK_Tegra_JP45_v3.5.5.run \ 
    && echo "# R32 (release), REVISION: 5.0, GCID: 25531747, BOARD: t186ref, EABI: aarch64, DATE: Fri Jan 15 23:21:05 UTC 2021" | tee /etc/nv_tegra_release \
    && /ZED_SDK_Tegra_JP45_v3.5.5.run -- silent 

# ************************************
# installing:
    # - cuda enabled pytorch version: 1.7.0 -> saved as 1.7.1
    # - mmcv version:  1.4.3
        # - mmdetection version:  2.19.0 
# ************************************
# Installing cuda enbaled pytorch  
WORKDIR /pytorch_wheel
COPY torch-1.7.1-cp37-cp37m-linux_aarch64.whl .
RUN pip3 install torch-1.7.1-cp37-cp37m-linux_aarch64.whl \
    && pip3 install torchvision==0.9.1 opencv_python mmdet==2.19.1 \ 
    && apt-get clean all \ 
    && rm -rf /var/lib/apt/lists/*

# Install cuda enabled mmcv (mmcv-full version)   
WORKDIR /mmcv_wheel 
COPY mmcv_full-1.4.3-cp37-cp37m-linux_aarch64.whl .
RUN pip3 install mmcv_full-1.4.3-cp37-cp37m-linux_aarch64.whl \
    && pip3 install pycocotools terminaltables rospkg \
    && apt-get clean all \ 
    && rm -rf /var/lib/apt/lists/*

# decord installation
WORKDIR /decord
# official PPA comes with ffmpeg 2.8, which lacks tons of features, we use ffmpeg 4.0 here
RUN sudo add-apt-repository ppa:jonathonf/ffmpeg-4 \
	&& sudo apt-get update \
	&& sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake \
	&& sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev \
	&& apt-get clean \
        && apt-get autoremove \
        && rm -rf /var/lib/apt/lists/* 

RUN git clone --recursive https://github.com/dmlc/decord
RUN cd decord \
	&& mkdir build && cd build \
	&& cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release \
	&& make \
	&& cd ../python \
	&& python3 setup.py install --user \
	&& apt-get clean \
        && apt-get autoremove \
        && rm -rf /var/lib/apt/lists/* 

# Done: install ros melodic with python 2
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# install ROS packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
		ros-melodic-${ROS_PKG} \
		ros-melodic-image-transport \
		ros-melodic-vision-msgs \
          python-rosdep \
          python-rosinstall \
          python-rosinstall-generator \
          python-wstool \
    && cd ${ROS_ROOT} \
    && rosdep init \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc

COPY zed_catkin_ws /workspace/zed_catkin_ws
WORKDIR /workspace/zed_catkin_ws

# build catkin workspace
RUN echo "Hello"  
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash ; catkin_make -DCMAKE_BUILD_TYPE=Release'
	
# source ros workspace
RUN echo 'source /workspace/zed_catkin_ws/devel/setup.bash' >> /root/.bashrc

# build MMaction2 from source and download model checkpoint files for humn detection and action recognition 
RUN cd src/mmdetection_ros/mmaction2 \
    && pip3 install -r requirements/build.txt \
    && pip3 install scipy==1.7.0 einops \
    && pip3 install -v -e . \
	&& pip3 install mmaction2 \
	&& pip3 install mmcv-full
   

# setup entrypoint
COPY ros_entrypoint.sh /workspace/zed_catkin_ws/ros_entrypoint.sh

ENTRYPOINT [ "./ros_entrypoint.sh" ]  

    
    
#    &&  wget -c https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
#        -O checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth
#    && wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
#        -O checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
#    && wget -c https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
#        -O checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
#    && wget -c https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth \
#        -O checkpoints/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth 
# COPY webcam_spatiotemporal.py src/mmdetection_ros/mmaction2/webcam_spatiotemporal.py 
