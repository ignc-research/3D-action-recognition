import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from gluoncv.utils import try_import_cv2
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

from gluoncv.utils import try_import_cv2

def tsn_ucf101(video_fname):


    '''video action recognition, e.g., use the same pre-trained model on an entire video.
    
    First, we download the video and sample the video frames at a speed of 1 frame per second.'''
    transform_fn = transforms.Compose([
        video.VideoCenterCrop(size=224),
        video.VideoToTensor(),
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    net = get_model('vgg16_ucf101', pretrained=True)

    cv2 = try_import_cv2()

    #url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/v_Basketball_g01_c01.avi'
    #video_fname = "Movie.mov"

    cap = cv2.VideoCapture(video_fname)
    cnt = 0
    video_frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        cnt += 1
        if ret and cnt % 25 == 0:
            video_frames.append(frame)
        if not ret: break

    cap.release()
    print('We evenly extract %d frames from the video %s.' % (len(video_frames), video_fname))

    if video_frames:
        video_frames_transformed = transform_fn(video_frames)
        final_pred = 0
        for _, frame_img in enumerate(video_frames_transformed):
            pred = net(nd.array(frame_img).expand_dims(axis=0))
            final_pred += pred
        final_pred /= len(video_frames)

        classes = net.classes
        topK = 5
        ind = nd.topk(final_pred, k=topK)[0].astype('int')
        print('The input video is classified to be')
        for i in range(topK):
            print('\t[%s], with probability %.3f.'%
                  (classes[ind[i].asscalar()], nd.softmax(final_pred)[0][ind[i]].asscalar()))



