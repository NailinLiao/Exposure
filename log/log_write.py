import csv
import socket
import os
import time
import pandas as pd


def buil_log_path(Func_name: str):
    User_RootPath = os.path.expanduser('~')
    # 获取本机ip
    save_path = os.path.join(User_RootPath, 'DeepWay', Func_name)
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)
    return save_path


def wirte_Check_log_file(img_file, statu, save_path=None):
    hostname = socket.gethostname()
    # 获取本机ip
    now_time = time.time()
    ip = str(socket.gethostbyname(hostname))
    if save_path != None:
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        end_path = os.path.join(save_path, str(ip) + '_overexposure_Check_Log' + '.csv')
    else:
        log_base_path = buil_log_path('overexposure_log')
        end_path = os.path.join(log_base_path, str(ip) + '_overexposure_Check_Log' + '.csv')
    if os.path.exists(end_path):
        with open(end_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([0, ip, img_file, str(statu), str(now_time)])
    else:
        row_data = {
            'ip': [ip],
            'img_file_name': [img_file],
            'statu': [statu],
            'time': [now_time],
        }
        DataFrame = pd.DataFrame(row_data)
        DataFrame.to_csv(end_path)


def wirte_Evaluate_log_file(row_dict, save_path=None):
    '''

    :param row_dict:
    :param save_path:
    :return:
    '''
    hostname = socket.gethostname()
    # 获取本机ip
    now_time = time.time()
    ip = str(socket.gethostbyname(hostname))
    if save_path != None:
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        end_path = os.path.join(save_path, str(ip) + '_overexposure_Evaluate_Log' + '.csv')
    else:
        log_base_path = buil_log_path('overexposure_log')
        end_path = os.path.join(log_base_path, str(ip) + '_overexposure_Evaluate_Log' + '.csv')
    if os.path.exists(end_path):
        with open(end_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([0, row_dict['acc'], row_dict['Precision'], row_dict['Recall'], row_dict['Fals_rat'],
                             row_dict['eval_log_file'], row_dict['predict_log_file'], ip, now_time])
    else:
        row_data = {
            'acc': [row_dict['acc']],
            'Precision': [row_dict['Precision']],
            'Recall': [row_dict['Recall']],
            'Fals_rat': [row_dict['Fals_rat']],
            'eval_log_file': [row_dict['eval_log_file']],
            'predict_log_file': [row_dict['predict_log_file']],
            'ip': [ip],
            'time': [now_time],
        }

        DataFrame = pd.DataFrame(row_data)
        DataFrame.to_csv(end_path)
