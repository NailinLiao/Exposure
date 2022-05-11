import os

import cv2
import pandas as pd
from log.log_write import wirte_Evaluate_log_file


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
        elif d_lss == 'csv':
            Csv_list.append(i)
        elif d_lss == 'jpg' or d_lss == 'png':
            imagelist.append(i)
    return jsonlist, imagelist, Csv_list


def evaluate_by_DataFram(pred_base_path: str, evalute_base_path: str, save_path=None):
    _, _, prea_list = get_input_path(pred_base_path)
    _, _, evalute_list = get_input_path(evalute_base_path)
    TP, FN, FP, TN = 0, 0, 0, 0,
    pred_DataFram = []
    evalute_DataFram = []
    for pred_DataFram_path in prea_list:
        if 'Check' in pred_DataFram_path:
            pred_DataFram.append(pd.read_csv(pred_DataFram_path))
    for evalute_DataFram_path in evalute_list:
        evalute_DataFram.append(pd.read_csv(evalute_DataFram_path))
    pred_DataFram = pd.concat(pred_DataFram)
    evalute_DataFram = pd.concat(evalute_DataFram)

    evaluate_fileNames = list(evalute_DataFram['file_name'])
    evaluate_labels = list(evalute_DataFram['statu'])
    pred_fileNames = list(pred_DataFram['img_file_name'])
    pred_labels = list(pred_DataFram['statu'])

    for i in range(len(evaluate_labels)):
        evaluate_label = evaluate_labels[i]
        evaluate_file_name = evaluate_fileNames[i]
        for index in range(len(pred_labels)):
            pred_label = pred_labels[index]
            pred_file_name = pred_fileNames[index]
            if os.path.split(pred_file_name)[-1] == os.path.split(evaluate_file_name)[-1]:
                if evaluate_label == pred_label:
                    if pred_label == 0:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if pred_label == 0:
                        FP += 1
                    else:
                        img = cv2.imread(pred_file_name)
                        cv2.imshow('img', img)
                        cv2.waitKey(0)
                        FN += 1
    N = (TP + TN + FN + FP)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Acc = (TP + TN) / N
    Fals_rat = (FN + FP) / N
    print('准确率：', Acc, '   精确度：', Precision, '   召回率:', Recall, '   错误率:', Fals_rat)
    test = {
        'acc': Acc,
        'Precision': Precision,
        'Recall': Recall,
        'Fals_rat': Fals_rat,
        'eval_log_file': evalute_list,
        'predict_log_file': prea_list,
    }
    wirte_Evaluate_log_file(test, save_path)

    return Acc, Precision, Recall


if __name__ == '__main__':
    pread_base_path = r'../log_file'
    evaluate_base_path = r'../evaluate/ValiaDataSet'
    save_path = r'../log_file'
    evaluate_by_DataFram(pread_base_path, evaluate_base_path, save_path)
