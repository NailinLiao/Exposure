# -*- coding: utf-8 -*-

import os
import cv2
from ReadImage import ReadImage
from Base_Func import get_input_path
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.models as models


def process_img(img_rgb):
    def img_transform(img_rgb, transform=None):
        """
        将数据转换为模型读取的形式
        :param img_rgb: PIL Image
        :param transform: torchvision.transform
        :return: tensor
        """
        img_rgb = cv2.resize(img_rgb, (224, 224))
        if transform is None:
            raise ValueError("找不到transform！必须有transform对img进行处理")

        img_t = transform(img_rgb)
        return img_t

    inference_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)  # chw --> bchw
    img_tensor = img_tensor.to('cpu')
    return img_tensor


class ExposureCheck:
    def __init__(self):
        pass

    @staticmethod
    def ThresholdInference(image_rgb, lowe=253, debug=False):
        '''
        阈值法推理图像过曝
        :param image_rgb:rgb通道的array图像
        :param lowe:亮度检测阈值 推荐参数240《 X 《 254
        :param debug:调试按钮
        :return:是否过曝，过曝面积占比
        '''
        img_ycr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YCR_CB)
        img_y = img_ycr[:, :, 0]
        h, w, _ = img_ycr.shape
        ret, thresh1 = cv2.threshold(img_y, lowe, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_cnts = []
        ALL_area = 0.00001
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            ALL_area += area
            if area > 5000:
                ALL_area += area
                big_cnts.append(cnt)
                image_rgb = cv2.drawContours(image_rgb, [cnt], -1, (0, 0, 0), 3)
        if debug:
            cv2.imshow('image_rgb', image_rgb)
        if len(big_cnts) != 0:
            return True, ALL_area / (h * w)
        else:
            return False, ALL_area / (h * w)

    @staticmethod
    def ModelInference(image_rgb, model_path):
        '''
        模型检测图像是否过曝
        :param image_rgb:rgb通道的array图像
        :param model_path:.pth 模型权重文件路径
        :return:是否过曝
        '''
        model = models.alexnet()
        num_ftrs = model.classifier._modules["6"].in_features
        model.classifier._modules["6"] = nn.Linear(num_ftrs, 2)
        pretrained_state_dict = torch.load(model_path)
        model.load_state_dict(pretrained_state_dict)
        model.eval()
        model.to('cpu')
        img_rgb_tensor = process_img(image_rgb)
        outputs = model(img_rgb_tensor)
        _, pred_int = torch.max(outputs.data, 1)
        label = pred_int.to('cpu').numpy()[0]
        if label == 0:
            return True
        else:
            return False
