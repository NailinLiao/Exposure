import os
import cv2
import random

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


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
        elif d_lss == 'jpg' or d_lss == 'png':
            imagelist.append(i)
    return jsonlist, imagelist, Csv_list


class DataSet():
    def __init__(self, input_path, size, transform):
        _, imagelist, _ = get_input_path(input_path)

        random.shuffle(imagelist)
        self.files = imagelist

        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = cv2.imread(self.files[item])
        img = cv2.resize(img, self.size, cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'YES' in self.files[item]:
            label = 0
        else:
            label = 1
        return self.transform(img), label


if __name__ == '__main__':
    input_path = r'D:/DataSet/KITTI/object/testing/image_2'
    dataset = DataSet(input_path, (512, 512), transform)
    print(len(dataset))
