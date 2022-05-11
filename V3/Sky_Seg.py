from model import U2NET  # full size version 173.6 MB
import torch
import numpy as np
from torchvision import transforms  # , utils
import os
from skimage import io, transform, color
from torch.autograd import Variable
from PIL import Image
import cv2


def get_input_path(input_path):
    '''
    检车路径下的所有文件
    :param input_path: path
    :param jsonlist: json文件列表
    :param imagelist: img文件列表
    :return:
    '''
    jsonlist = []
    imagelist = []
    Csv_list = []

    def get_file_path(root_path, file_list):
        '''
        获取跟文件下的所有文件 包括子文件夹中的文件保存于
        file——list中

        :param root_path: 需要获取文件的根目录
        :param file_list: 保存文件列表
        :return: 空
        '''
        dir_or_files = os.listdir(root_path)
        for dir_file in dir_or_files:
            dir_file_path = os.path.join(root_path, dir_file)
            if os.path.isdir(dir_file_path):
                #                 pass
                #                 pass
                get_file_path(dir_file_path, file_list)
            else:
                file_list.append(dir_file_path)

    file_list = []
    get_file_path(input_path, file_list)

    for i in file_list:
        d_lss = os.path.split(i)[-1].split('.')[-1]
        if d_lss == 'json':
            jsonlist.append(i)
        elif d_lss == 'csv' or d_lss == 'xlsx':
            Csv_list.append(i)
        elif d_lss == 'jpg' or d_lss == 'png' or d_lss == 'jpeg':
            imagelist.append(i)
    return jsonlist, imagelist, Csv_list


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        image = sample

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpImg = tmpImg.transpose((2, 0, 1))

        return torch.from_numpy(tmpImg)


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')

        return img


class SkySeg:
    def __init__(self, pth_path: str, cuda: bool = True):
        self.net = U2NET(3, 1)
        if cuda:
            self.net.load_state_dict(torch.load(pth_path))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(pth_path, map_location='cpu'))
        self.net.eval()
        self.transform = transforms.Compose([RescaleT(320),
                                             ToTensorLab(flag=0)])

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    def inference(self, img_rgb):
        # image = io.imread(self.image_name_list[idx])
        image = img_rgb
        if 2 == len(image.shape):
            image = image[:, :, np.newaxis]
        sample = self.transform(image)

        sample = sample.type(torch.FloatTensor)
        sample = sample.unsqueeze(0)
        inputs_test = Variable(sample.cuda())
        d1, d2, d3, d4, d5, d6, d7 = self.net(inputs_test)
        # normalization
        pred = d1[:, 0, :, :]
        pred = self.normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np * 255).convert('RGB')

        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

        return np.array(imo)


def seg_sky(img_file_path, model_path, Debug=True):
    clea_rat = 0.05
    skySeg = SkySeg(model_path)
    img = cv2.imread(img_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = skySeg.inference(img)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    h, w, _ = img.shape
    clea_area = h * w * clea_rat

    contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret = []
    for cnt in contours:
        if cv2.contourArea(cnt) > clea_area:
            ret.append(cnt)
    if Debug:
        cv2.fillPoly(img, ret, (0, 0, 0))
        cv2.imshow('img', img)
        cv2.imshow('mask', out)
        cv2.waitKey(0)
    return ret


if __name__ == '__main__':
    import cv2

    image_base_path = r'../DataSet/True'
    u2net_pth = r'E:\WorkSpace\Exposure\model\u2net_bce_itr_26000_train_0.106366_tar_0.009548.pth'
    save_path = r'./ret/'
    _, image_list, _ = get_input_path(image_base_path)
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    for index, image_file in enumerate(image_list):
        seg_sky(image_file, u2net_pth)
        # img = cv2.imread(image_file)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # out = 255 - skySeg.inference(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.bitwise_and(img, out)
        # end_path = os.path.join(save_path, str(index) + '.png')
        # cv2.imwrite(end_path, img)
