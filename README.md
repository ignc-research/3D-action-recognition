# 3D-action-recognition

This repository implements two demos:  
    - a real-time spatio-temporal action detection from a web camera as by MMACTION2
    - a single person action recognition from a webcamera as by MMACTION2,
and integrates them with ROS.


## Docker 

1. Build: sudo docker build -t hichdh/mmaction:2.5.1 . 
2. Run: sudo docker run -it --runtime nvidia --gpus all --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix:rw hichdh/mmaction:2.5.1 /bin/bash


## Webcam demo

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--camera-id ${CAMERA_ID}] [--threshold ${THRESHOLD}] \
    [--average-size ${AVERAGE_SIZE}] [--drawing-fps ${DRAWING_FPS}] [--inference-fps ${INFERENCE_FPS}]
```

Optional arguments:

- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `CAMERA_ID`: ID of camera device If not specified, it will be set to 0.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.
- `AVERAGE_SIZE`: Number of latest clips to be averaged for prediction. If not specified, it will be set to 1.
- `DRAWING_FPS`: Upper bound FPS value of the output drawing. If not specified, it will be set to 20.
- `INFERENCE_FPS`: Upper bound FPS value of the output drawing. If not specified, it will be set to 4.

:::{note}
If your hardware is good enough, increasing the value of `DRAWING_FPS` and `INFERENCE_FPS` will get a better experience.
:::

Examples: 

Assume that you are located at `$MMACTION2` 

1. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth tools/data/kinetics/label_map_k400.txt --average-size 5 \
      --threshold 0.2 --device cpu
    ```

2. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      tools/data/kinetics/label_map_k400.txt --average-size 5 --threshold 0.2 --device cpu
    ```

3. Recognize the action from web camera as input by using a I3D model on gpu by default, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth tools/data/kinetics/label_map_k400.txt \
      --average-size 5 --threshold 0.2
    ```

:::{note}
Considering the efficiency difference for users' hardware, Some modifications might be done to suit the case.
Users can change:

1). `SampleFrames` step (especially the number of `clip_len` and `num_clips`) of `test_pipeline` in the config file, like `--cfg-options data.test.pipeline.0.num_clips=3`.
2). Change to the suitable Crop methods like `TenCrop`, `ThreeCrop`, `CenterCrop`, etc. in `test_pipeline` of the config file, like `--cfg-options data.test.pipeline.4.type=CenterCrop`.
3). Change the number of `--average-size`. The smaller, the faster.
:::

## SpatioTemporal Action Detection Webcam Demo

```shell
python demo/webcam_demo_spatiotemporal_det.py \
    [--config ${SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--checkpoint ${SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRESHOLD}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--input-video] ${INPUT_VIDEO} \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--output-fps ${OUTPUT_FPS}] \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--show] \
    [--display-height] ${DISPLAY_HEIGHT} \
    [--display-width] ${DISPLAY_WIDTH} \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--clip-vis-length] ${CLIP_VIS_LENGTH}
```

Optional arguments:

- `SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE`: The spatiotemporal action detection config file path.
- `SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT`: The spatiotemporal action detection checkpoint path or URL.
- `ACTION_DETECTION_SCORE_THRESHOLD`: The score threshold for action detection. Default: 0.4.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_DETECTION_SCORE_THRE`: The score threshold for human detection. Default: 0.9.
- `INPUT_VIDEO`: The webcam id or video path of the source. Default: `0`.
- `LABEL_MAP`: The label map used. Default: `tools/data/ava/label_map.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Default: `cuda:0`.
- `OUTPUT_FPS`: The FPS of demo video output. Default: 15.
- `OUTPUT_FILENAME`: Path to the output file which is a video format. Default: None.
- `--show`: Whether to show predictions with `cv2.imshow`.
- `DISPLAY_HEIGHT`: The height of the display frame. Default: 0.
- `DISPLAY_WIDTH`: The width of the display frame. Default: 0. If `DISPLAY_HEIGHT <= 0 and DISPLAY_WIDTH <= 0`, the display frame and input video share the same shape.
- `PREDICT_STEPSIZE`: Make a prediction per N frames. Default: 8.
- `CLIP_VIS_LENGTH`: The number of the draw frames for each clip. In other words, for each clip, there are at most `CLIP_VIS_LENGTH` frames to be draw around the keyframe. DEFAULT: 8.

Tips to get a better experience for webcam demo:

- How to choose `--output-fps`?

  - `--output-fps` should be almost equal to read thread fps.
  - Read thread fps is printed by logger in format `DEBUG:__main__:Read Thread: {duration} ms, {fps} fps`

- How to choose `--predict-stepsize`?

  - It's related to how to choose human detector and spatio-temporval model.
  - Overall, the duration of read thread for each task should be greater equal to that of model inference.
  - The durations for read/inference are both printed by logger.
  - Larger `--predict-stepsize` leads to larger duration for read thread.
  - In order to fully take the advantage of computation resources, decrease the value of `--predict-stepsize`.

Examples:

Assume that you are located at `$MMACTION2`. Use the Faster RCNN as the human detector, SlowOnly-8x8-R101 as the action detector. Making predictions per 40 frames, and FPS of the output is 20. Show predictions with `cv2.imshow`.

```shell
python demo/webcam_demo_spatiotemporal_det.py \
    --input-video 0 \
    --config configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 40 \
    --output-fps 20 \
    --show
```
#Tips
- If docker access problems submerge: Run 'xhost +'
- When pulling the repository, use 'git lfs pull'
