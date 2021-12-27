import time

import gluoncv as gcv
from gluoncv.utils import try_import_cv2

cv2 = try_import_cv2()
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

from gluoncv.utils.filesystem import try_import_decord
from decord import VideoReader
decord = try_import_decord()


if __name__ == '__main__':

    # Load the model
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
    # Compile the model for faster speed
    net.hybridize()
    # Load the webcam handler
    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus

    axes = None
    NUM_FRAMES = 2000 # you can change this
    for i in range(NUM_FRAMES):
        # Load frame from the camera
        ret, frame = cap.read()

        # Image pre-processing
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

        # Run frame through network
        class_IDs, scores, bounding_boxes = net(rgb_nd)
        labels=[]
        prob=[]


        img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
        gcv.utils.viz.cv_plot_image(img)

        for j in range(len(bounding_boxes[0])):

            if class_IDs[0][j] == 14.:
                 # do action recognition
                transform_fn = transforms.Compose([
                    video.VideoCenterCrop(size=224),
                    video.VideoToTensor(),
                    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_list = transform_fn([frame])
                #plt.imshow(np.transpose(img_list[0], (1, 2, 0)))
                #plt.show()

                net = get_model('vgg16_ucf101', nclass=101, pretrained=True)
                pred = net(nd.array(img_list[0]).expand_dims(axis=0))


                classes = net.classes
                topK = 1
                ind = nd.topk(pred, k=topK)[0].astype('int')
                #print('The input video frame is classified to be')
                for i in range(topK):
                     #print('\t[%s], with probability %.3f.' %
                     #      (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
                    #labels.append(classes[ind[i].asscalar()])


                    #prob.append(nd.softmax(pred)[0][ind[i]].asscalar())
                    p=np.asarray([nd.softmax(pred)[0][ind[i]].asscalar()])
                    l = np.asarray([ind[i].asscalar()])





                img = gcv.utils.viz.cv_plot_bbox(frame,[bounding_boxes[0][j]], p, l,
                                                  class_names=net.classes)

                gcv.utils.viz.cv_plot_image(img)



                # Display the result

                labels=np.array(labels)
                prob=np.array(prob)
                #img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], prob, labels, class_names=net.classes)

                img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)

                gcv.utils.viz.cv_plot_image(img)
                cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
