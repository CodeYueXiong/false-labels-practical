# ----------------------------------------------------#
#   get the detection results with test set
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
import colorsys
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from PIL import Image
from tqdm import tqdm

from nets.ssd import get_ssd
from nets.ssd import SSD
from utils.box_utils import letterbox_image, ssd_correct_boxes

MEANS = (104, 117, 123)
global model
model = get_ssd("test", 5, "vgg", 0.01, 0.45)

'''
The lower threshold set here is because the Recall and Precision values under different threshold conditions are needed to calculate the map.
Therefore, the calculated map will be more accurate only if there are enough boxes reserved. For details, you can understand the principle of map.
The Recall and Precision values output when calculating the map refer to the Recall and Precision values when the threshold is 0.5.
The number of txt boxes in ./input/detection-results/ obtained here will be a bit more than that of direct predict. This is because the threshold here is low.
The purpose is to calculate the Recall and Precision values under different threshold conditions, so as to realize the calculation of the map.
Some students may know that there are 0.5 and 0.5:0.95 mAPs.
If you want to set mAP0.x, such as mAP0.75, you can go to get_map.py to set MINOVERLAP.
'''


class mAP_SSD(SSD):
    def generate(self):
        self.confidence = 0.01
        # -------------------------------#
        #   Calculate the total number of classes
        # -------------------------------#
        self.num_classes = len(self.class_names) + 1

        # -------------------------------#
        #   Load model and weights
        # -------------------------------#
        model = get_ssd("test", self.num_classes, self.backbone, self.confidence, self.nms_iou)
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # Set different colors for the picture frame
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   Detection pictures
    # ---------------------------------------------------#
    def detect_image(self, image_id, image):
        # self.confidence = 0.01
        self.class_names = ["Blob", "Blur", "Distortion", "Channel_Change"]
        # -------------------------------#
        #   Calculate the total number of classes
        # -------------------------------#
        self.num_classes = len(self.class_names) + 1
        self.backbone = "vgg"
        self.confidence, self.nms_iou = 0.01, 0.45

        # -------------------------------#
        #   Load model and weights
        # -------------------------------#
        # model = get_ssd("test", self.num_classes, self.backbone, self.confidence, self.nms_iou)  # global model
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = "/content/drive/MyDrive/falselabels/data/logs_ssd300/Epoch100-Total_Loss1.4187-Val_Loss0.9743.pth"
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        self.cuda = True
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # Set different colors for the picture frame
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.letterbox_image = True
        self.input_shape = 300, 300
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        image_shape = np.array(np.shape(image)[0:2])
        print("image_shape:\n", image_shape[0])

        # ---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for detection
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.input_shape[1], self.input_shape[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.input_shape[1], self.input_shape[0]),
                                       Image.BICUBIC)  # self.input_shape[1],self.input_shape[0]

        photo = np.array(crop_img, dtype=np.float64)
        with torch.no_grad():
            photo = torch.from_numpy(np.expand_dims(np.transpose(photo - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            preds = self.net(photo)

            top_conf = []
            top_label = []
            top_bboxes = []
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.confidence:
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i - 1]
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        if len(top_conf) <= 0:
            return

        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
        # -----------------------------------------------------------#
        #   Remove the gray bars
        # -----------------------------------------------------------#
        if self.letterbox_image:
            boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                      np.array([self.input_shape[0], self.input_shape[1]]),
                                      image_shape)  # self.input_shape[0],self.input_shape[1]
        else:
            top_xmin = top_xmin * image_shape[1]
            top_ymin = top_ymin * image_shape[0]
            top_xmax = top_xmax * image_shape[1]
            top_ymax = top_ymax * image_shape[0]
            boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        for i, c in enumerate(top_label):
            predicted_class = c
            score = str(float(top_conf[i]))

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

# get ssd attributes
def get_ssd_attr(backbone_name="vgg"):
    num_classes = 5
    if backbone_name == 'vgg':
        backbone, extra_layers = vgg(3), add_extras(1024, backbone_name)
        mbox = [4, 6, 6, 6, 4, 4]
    else:
        backbone, extra_layers = mobilenet_v2().features, add_extras(1280, backbone_name)
        mbox = [6, 6, 6, 6, 6, 6]

    loc_layers = []
    conf_layers = []

    if backbone_name == 'vgg':
        backbone_source = [21, -2]
        # ---------------------------------------------------#
        #   在add_vgg获得的特征层里
        #   第21层和-2层可以用来进行回归预测和分类预测。
        #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
        # ---------------------------------------------------#
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                     mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                                      mbox[k] * num_classes, kernel_size=3, padding=1)]
        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                      * num_classes, kernel_size=3, padding=1)]
    else:
        backbone_source = [13, -1]
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                     mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                                      mbox[k] * num_classes, kernel_size=3, padding=1)]

        for k, v in enumerate(extra_layers, 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                      * num_classes, kernel_size=3, padding=1)]
    return backbone, extra_layers, loc_layers, conf_layers


if __name__ == "__main__":
    backbone, extra_layers, loc_layers, conf_layers = get_ssd_attr(backbone_name="vgg")

    ssd = mAP_SSD(phase="test",
                  base=backbone,
                  extras=extra_layers,
                  head=(loc_layers, conf_layers),
                  num_classes=5,
                  confidence=0.01,
                  nms_iou=0.45,
                  backbone_name="vgg")

    image_ids = open('./ImageSets/Main/test.txt').read().strip().split()
    print(image_ids)
    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    for image_id in tqdm(image_ids):
        image_path = "Test/" + image_id + ".png"
        image = Image.open(image_path)
        # When triggered, the calculation of mAP can be visualized afterwards
        # image.save("./input/images-optional/"+image_id+".jpg")
        ssd.detect_image(image_id, image)

    print("Conversion completed!")