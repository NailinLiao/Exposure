import os
import pandas as pd


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


def main(Validation_path, save_path):
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)
    overexposure_path = os.path.join(Validation_path, 'overexposure')
    normal_path = os.path.join(Validation_path, 'normal')
    _, overexposure_images, _ = get_input_path(overexposure_path)
    _, normal_images, _ = get_input_path(normal_path)
    over = {
        'file_name': overexposure_images,
        'statu': [1] * len(overexposure_images)
    }

    over_DataFram = pd.DataFrame(over)

    normal = {
        'file_name': normal_images,
        'statu': [0] * len(normal_images)
    }
    normal_DataFram = pd.DataFrame(normal)
    ALL_DataFram = pd.concat([over_DataFram, normal_DataFram])
    end_path = os.path.join(save_path, 'ValidationDataSet.csv')
    print(end_path)
    ALL_DataFram.to_csv(end_path)


if __name__ == '__main__':
    Validation_path = r'../DataSet'
    save_path = r'./ValiaDataSet'
    main(Validation_path, save_path)
    # print([0] * 100)
