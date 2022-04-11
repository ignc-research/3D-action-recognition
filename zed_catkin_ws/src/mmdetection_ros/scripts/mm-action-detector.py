#!/usr/bin/env python3
"""
 @Author: Hichem Dhouib
 @Date: 2022
 @Last Modified by:   Hichem Dhouib & Seif Daknou
 @Last Modified time:
"""

import sys
import cv2
import numpy as np
import os
from cv_bridge import CvBridge
from mmdet.apis import inference_detector, init_detector
import rospy
import message_filters
from sensor_msgs.msg import Image , CompressedImage , CameraInfo
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from logging import debug
from time import time
from contextlib import contextmanager
from funcy import print_durations
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from zed_interfaces.msg import ObjectsStamped, BoundingBox2Df,  BoundingBox3D

import mmdet
import mmcv
import torch
import pdb

import mmaction
from mmaction.models import build_detector
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint


rospy.logdebug("Start MMdetecctor Action Recognition")

CONFIG_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/mmaction2/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py'
MODEL_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/mmaction2/checkpoints/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217-ae225e97.pth'
LABEL_MAP_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/mmaction2/tools/data/ava/label_map.txt'
COLORS_2D = (0,0,255)
SCALE =  1
COLORS_MARKER_3D = [1, 0 , 0 , 0.3]
COLORS_TEXT_3D = [0.1, 0.8 , 0.1 , 1]
DELETEALL_MARKER_ID = 300
MARKER_3D_COUNTER_ID = 150
MARKER_TEXT_COUNTER_ID = 0

@contextmanager
def timer(descrption: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start
    rospy.logdebug(f"{descrption}: {ellapsed_time}")

def load_label_map(file_path):
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

def delete_markers():
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.action = Marker.DELETEALL
    marker.id = DELETEALL_MARKER_ID
    return marker

def make_text_marker(marker_location):
    marker = Marker()
    marker.header.frame_id =  "base_link"
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.scale.x = SCALE/10
    marker.scale.y = SCALE/10
    marker.scale.z = SCALE/10
    marker.header.stamp  = rospy.get_rostime()
    marker.id = MARKER_TEXT_COUNTER_ID +1
    marker.pose.position.y, marker.pose.position.z , marker.pose.position.x = marker_location
    marker.pose.position.z = - marker.pose.position.z * 10
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = COLORS_TEXT_3D
    return marker

def make_3d_marker(marker_location):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.scale.x = SCALE / 2
    marker.scale.y = SCALE / 2
    marker.scale.z = SCALE
    marker.header.stamp = rospy.get_rostime()
    marker.id = MARKER_3D_COUNTER_ID +1
    marker.pose.position.y, marker.pose.position.z, marker.pose.position.x = marker_location
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = COLORS_MARKER_3D
    return marker

class ActionDetector:
    def __init__(self, model, device, config):

        self.bridge = CvBridge()

        # 2D
        self.pub_topic_color = "/mmdet/pose_estimation/det2d/compressed"
        self.image_pub = rospy.Publisher(self.pub_topic_color, CompressedImage, queue_size=1)

        # 3D
        self.pub_topic_marker_array = "/mmdet/visualization_marker_array"
        self.marker_array_pub = rospy.Publisher(self.pub_topic_marker_array, MarkerArray, queue_size=1)

        # Subscribe to ZED 2 object detection data
        self.sub_topic_det_results_data = "/zed2/zed_node/obj_det/objects"
        self.hum_det_results_sub = message_filters.Subscriber(self.sub_topic_det_results_data, ObjectsStamped)

        # Subscribe to ZED 2 rectified RGB image
        self.sub_topic_color = "/zed2/zed_node/rgb/image_rect_color"
        self.rgb_image_sub = message_filters.Subscriber(self.sub_topic_color, Image)

        # Visualization parameters in RVIZ
        self.visualization_3d = rospy.get_param("visualization_3d")
        self.visualization_2d = rospy.get_param("visualization_2d")

        self.compressed_cv_image = None
        self.compressed_cv_images_list = []
        self.marker_array_msg = MarkerArray()

        # Action detection model from MMaction
        self.model = model
        self.config = config
        self.device = torch.device(device)

        # stdet sampling strategy (fron neural network config)
        val_pipeline = self.config.data.val.pipeline
        sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
        self.clip_len, self.frame_interval = sampler['clip_len'], sampler['frame_interval']
        self.predict_stepsize = 40
        self.window_size = self.clip_len * self.frame_interval
        self.buffer_size = self.window_size - self.predict_stepsize
        frame_start = self.window_size // 2 - (self.clip_len // 2) * self.frame_interval
        self.frames_inds = [frame_start + 1 +  self.frame_interval * i for i in range(self.clip_len)]
        assert self.clip_len % 2 == 0, 'We would like to have an even clip_len'

        self.stdet_input_shortside=256

        img_norm_cfg = config['img_norm_cfg']
        if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
            to_bgr = img_norm_cfg.pop('to_bgr')
            img_norm_cfg['to_rgb'] = to_bgr
        img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
        img_norm_cfg['std'] = np.array(img_norm_cfg['std'])
        self.img_norm_cfg = img_norm_cfg

        # Load label_map
        self.label_map = load_label_map(LABEL_MAP_PATH)
        self.new_h = 0
        self.new_w = 0
        self.w_ratio = 0
        self.h_ratio = 0

        # helper params for buffer and slowonly input data
        self.detected_humans_proposals = []
        self.bbox = []
        self.bboxes_proposal_frame = []
        self.bboxes_proposal_clip = []
        self.clip = []
        self.images_metas = []
        self.clip_frames_counter = 0
        self.action_score_thr = 0.4
        self.control_idx = 0
        self.inference_number = 0
        self.marker_3d_list = []
        self.marker_text_list = []
        self.marker_counter_id = 0

    def pre_vis_2d(self, image_np, kp0, kp2):
        start_point = (int(kp0[0] * self.w_ratio) ,int(kp0[1] * self.h_ratio))
        end_point = (int(kp2[0] * self.w_ratio) ,int(kp2[1] * self.h_ratio))
        cv_img = cv2.rectangle(image_np, start_point, end_point, COLORS_2D, 3)
        self.compressed_cv_image = self.bridge.cv2_to_compressed_imgmsg(cv_img)
        self.compressed_cv_images_list.append(self.compressed_cv_image)

    #extract center, height and width of 2d human bbox.
    def zed2_2d_bbox_preprocessing(self, subResult, kp0, kp2):
        y = int(subResult[0][0] / 2)
        x = int(subResult[0][1] / 2) 
        width = int((subResult[2][0] - subResult[0][0]))
        height = int((subResult[2][1] - subResult[0][1]))
        return [x,y,width,height]

    def image_preprocessing(self, image):
        #rospy.logdebug("w = %s / h = %s", image.width, image.height)
        image_np = np.frombuffer(image.data, dtype = np.uint8).reshape(image.height, image.width, -1)

        custom_width = int(376 / 2)
        custom_height = int(1344 / 8)
        image_np = cv2.resize(image_np, (custom_width,custom_height))

        self.new_w, self.new_h = mmcv.rescale_size((custom_width, custom_height), (self.stdet_input_shortside, np.Inf))
        #rospy.logdebug("new_w = %s / new_h = %s", self.new_w, self.new_h)
        image_np = mmcv.imresize(image_np, (self.new_w, self.new_h))

        self.w_ratio, self.h_ratio = self.new_w / image.width , self.new_h / image.height
        #rospy.logdebug(" w_ratio | h_ratio: %s, %s", self.w_ratio, self.h_ratio)
        return image_np

    def processing_3d_bboxes(self, bbox_3d_corners):
        # 3D bounding box
        bbox_3d_pixel_0 = bbox_3d_corners.corners[0]
        bbox_3d_pixel_1 = bbox_3d_corners.corners[1]
        bbox_3d_pixel_2 = bbox_3d_corners.corners[2]
        bbox_3d_pixel_3 = bbox_3d_corners.corners[3]
        bbox_3d_pixel_4 = bbox_3d_corners.corners[4]
        bbox_3d_pixel_5 = bbox_3d_corners.corners[5]
        bbox_3d_pixel_6 = bbox_3d_corners.corners[6]
        bbox_3d_pixel_7 = bbox_3d_corners.corners[7]

        x_center = (bbox_3d_pixel_1.kp[0] + bbox_3d_pixel_7.kp[0]) / 2
        y_center = (bbox_3d_pixel_1.kp[1] + bbox_3d_pixel_7.kp[1]) / 2
        z_center = (bbox_3d_pixel_1.kp[2] + bbox_3d_pixel_7.kp[2]) / 2

        bbox_3d = [y_center, z_center, x_center]
        return bbox_3d

    def processing_human_detection_results(self, image_np, objects_stamped):
        self.bboxes_proposal_frame = []
        for counter, detected_human in enumerate(objects_stamped.objects):
            # 2D bounding box
            bbox_2d_pixel_0 = detected_human.bounding_box_2d.corners[0]
            bbox_2d_pixel_1 = detected_human.bounding_box_2d.corners[1]
            bbox_2d_pixel_2 = detected_human.bounding_box_2d.corners[2]
            bbox_2d_pixel_3 = detected_human.bounding_box_2d.corners[3]

            self.bbox = [bbox_2d_pixel_0.kp, bbox_2d_pixel_1.kp, bbox_2d_pixel_2.kp, bbox_2d_pixel_3.kp]
            rospy.logdebug("appending 3d bbox and text marker: %s", counter)
            self.bbox = self.zed2_2d_bbox_preprocessing(self.bbox, bbox_2d_pixel_0.kp, bbox_2d_pixel_2.kp)
            self.bbox = np.array(self.bbox, dtype=np.float32)

            self.bbox[0] = self.bbox[0] * self.w_ratio
            self.bbox[1] = self.bbox[1] * self.h_ratio
            self.bbox[2] = self.bbox[2] * self.w_ratio
            self.bbox[3] = self.bbox[3] * self.h_ratio

            self.bboxes_proposal_frame.append(self.bbox)
            if self.visualization_2d : self.pre_vis_2d(image_np, bbox_2d_pixel_0.kp, bbox_2d_pixel_2.kp)

            bbox_3d = self.processing_3d_bboxes(detected_human.bounding_box_3d)
            marker_3d = make_3d_marker(bbox_3d)
            marker_text = make_text_marker(bbox_3d)
            self.marker_3d_list.append(marker_3d)
            self.marker_text_list.append(marker_text)

        tensor_bboxes_frame_proposal = torch.from_numpy(np.asarray(self.bboxes_proposal_frame)).to(self.device)
        self.bboxes_proposal_clip.append(tensor_bboxes_frame_proposal)

        # Append converted values to float32, normalized and resized images
        image_np = image_np.astype(np.float32)
        image_np = image_np[:,:,:3]
        image_np = mmcv.image.imnormalize(image_np, self.img_norm_cfg['mean'], self.img_norm_cfg['std'])

        self.clip.append(image_np)
        self.clip_frames_counter +=1

    def action_recogntion(self):

        return_loss = False
        # Stacked processed BGR Frames - 2nd parameter: THWC -> CTHW -> 1CTHW
        input_array = np.stack(self.clip).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(self.device)

        # 3rd parameter: Dict of correctd height and width
        self.images_metas = dict(img_shape=(self.new_h, self.new_w))

        self.image_pub.publish(self.compressed_cv_images_list[-1])

        self.inference_number += 1
        with torch.no_grad():
            result = self.model(
                return_loss=False,
                img=[input_tensor],
                img_metas=[[self.images_metas]],
                proposals=[[self.bboxes_proposal_clip]]
                )[0]

        proposal = self.bboxes_proposal_clip[-1]

        # Perform action score thr
        for class_id in range(len(result)):
            if class_id + 1 not in self.label_map: continue
            for bbox_id in range(proposal.shape[0]):
                if result[class_id][bbox_id, 4] > self.action_score_thr:
                    rospy.logdebug("INF: %s : Iterating for bbox_id %s over AR results: %s", self.inference_number ,bbox_id ,(self.label_map[class_id + 1], result[class_id][bbox_id,4]))
                    predicted_label = self.label_map[class_id + 1]
                    prediction_score = result[class_id][bbox_id,4]
                    self.marker_text_list[bbox_id].text = self.marker_text_list[bbox_id].text + """
                    """ + predicted_label + ' : ' + str(prediction_score)

        for bbox_id in range(proposal.shape[0]):
            rospy.logdebug("proposal.shape[0]: %s | bbox_id in marker array: %s", proposal.shape[0] ,bbox_id)
            self.marker_array_msg.markers.append(self.marker_3d_list[bbox_id])
            self.marker_array_msg.markers.append(self.marker_text_list[bbox_id])

    def reset(self):
        self.clip_frames_counter = 0
        self.bbox = []
        self.bboxes_proposal_frame = []
        self.bboxes_proposal_clip = []
        self.clip = []
        self.images_metas = []
        self.detected_humans_proposals = []
        self.compressed_cv_images_list = []
        self.control_idx = 0
        self.marker_3d_list = []
        self.marker_text_list = []
        DELETEALL_MARKER_ID = 300
        MARKER_3D_COUNTER_ID = 150

    def callback(self, image, objects_stamped):
        if objects_stamped.objects == []:
            # 'We would like to process only frames with detected huamns'
            return

        if self.control_idx in self.frames_inds:
            image_np = self.image_preprocessing(image)
            self.processing_human_detection_results(image_np, objects_stamped)
            if self.clip_frames_counter == self.buffer_size:
                self.action_recogntion()
                self.marker_array_pub.publish(self.marker_array_msg.markers)
                self.reset()
        self.control_idx += 1

def main():

    config = Config.fromfile(CONFIG_PATH)
    checkpoint = MODEL_PATH
    config.model.backbone.pretrained = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    _ = load_checkpoint(model, checkpoint, map_location='cpu')

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    detector = ActionDetector(model, device, config)

    rospy.init_node('mmdetector', log_level=rospy.DEBUG)
    ts = message_filters.ApproximateTimeSynchronizer([detector.rgb_image_sub, detector.hum_det_results_sub], queue_size=1, slop=0.6, allow_headerless=True)
    ts.registerCallback(detector.callback)
    rospy.spin()


if __name__=='__main__':
    main()