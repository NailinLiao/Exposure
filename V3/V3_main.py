'''
脚本内容:
    实现批量\单张过曝检测
        - 并读写检测日志
    更新:
        V2.01 增加 对天空的 分割模型 ,将天空进行掩盖 使用阈值判断过曝
            时间:20220325


'''
import os
import json

import cv2

from Sky_Seg import *
from Highlighted_Seg import seg_Highlighted_area
from Fuzzy_Seg import *





def clead_area_function(image_file, sky_cnts, Highlighted_cnts, Fuzzy_cnts, Debug=True):
    img = cv2.imread(image_file)
    h, w, _ = img.shape
    mask = np.zeros((h, w))
    seg_mask = mask.copy()
    Highlighted_mask = mask.copy()
    Fuzzy_mask = mask.copy()

    seg_mask = cv2.fillPoly(seg_mask, sky_cnts, 255)
    Highlighted_mask = cv2.fillPoly(Highlighted_mask, Highlighted_cnts, 255)
    Fuzzy_mask = cv2.fillPoly(Fuzzy_mask, Fuzzy_cnts, 255)

    Fuzzy_mask[seg_mask != 0] = 0
    Fuzzy_mask[Highlighted_mask != 0] = 0

    Highlighted_mask[seg_mask != 0] = 0

    # cv2.imshow('a', seg_mask)
    # cv2.imshow('b', Highlighted_mask)
    # cv2.imshow('c', Fuzzy_mask)
    # cv2.waitKey(0)
    Fuzzy_cnts, _ = cv2.findContours(np.array(Fuzzy_mask, dtype='uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Highlighted_cnts, _ = cv2.findContours(np.array(Highlighted_mask, dtype='uint8'), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    img[:, :, 0][seg_mask != 0] += 100
    img[:, :, 1][Fuzzy_mask != 0] += 100
    img[:, :, 2][Highlighted_mask != 0] += 100

    if False:
        cv2.imshow('seg_mask', seg_mask)
        cv2.imshow('Highlighted_mask', Highlighted_mask)
        cv2.imshow('Fuzzy_mask', Fuzzy_mask)
        cv2.waitKey(0)
    return sky_cnts, Fuzzy_cnts, Highlighted_cnts, img


def scoring_image(image_file, sky_cnts, Fuzzy_cnts, Highlighted_cnts):
    img = cv2.imread(image_file)
    h, w, _ = img.shape
    count = 0
    for Highlight in Highlighted_cnts:
        count += cv2.contourArea(Highlight)
    return str(count / (h * w))


def funstion_post_processing(image_file: str, sky_cnts, Fuzzy_cnts, Highlighted_cnts, img_ret, save_path):
    base_path = os.path.split(image_file)[0]

    base_path = os.path.join(save_path, str(base_path).split(':')[-1][1:])

    file_name = str(os.path.split(image_file)[-1]).split('.')[0]

    cnts = []
    for sky in sky_cnts:
        cnts.append({
            'sky': np.array(sky).tolist(),
            'targ_type': 'polygon',
        })

    for Fuzzy in Fuzzy_cnts:
        cnts.append({
            'Fuzzy': np.array(Fuzzy).tolist(),
            'targ_type': 'polygon',
        })

    for Highlighted in Highlighted_cnts:
        cnts.append({
            'Highlighted': np.array(Highlighted).tolist(),
            'targ_type': 'polygon',
        })

    json_dict = {
        'file_name': str(file_name),
        'target_cnts': list(cnts),
        'score': scoring_image(image_file, sky_cnts, Fuzzy_cnts, Highlighted_cnts),
    }

    score_path = os.path.join(base_path, 'score_path')
    jaon_path = os.path.join(score_path, 'json')
    png_path = os.path.join(score_path, 'png')
    if os.path.exists(score_path):
        pass
    else:
        os.makedirs(score_path)
        os.makedirs(jaon_path)
        os.makedirs(png_path)
    json_file_path = os.path.join(jaon_path, file_name + '.json')
    png_file_path = os.path.join(png_path, file_name + '.png')
    with open(json_file_path, "w") as json_file:
        json.dump(json_dict, json_file)
        print(file_name, '       ______________Write_______json____________OK!')
    cv2.imwrite(png_file_path, img_ret)
    print(file_name, '       ______________Write_______img_ret____________OK!')


def main(image_file, U2net_path, save_path):
    sky_cnts = seg_sky(image_file, U2net_path, Debug=False)
    Highlighted_cnts = seg_Highlighted_area(image_file, lowe=253, Debug=False)
    Fuzzy_cnts = seg_Fuzzy(image_file, Debug=False)
    sky_cnts, Fuzzy_cnts, Highlighted_cnts, img_ret = clead_area_function(image_file, sky_cnts, Highlighted_cnts,
                                                                          Fuzzy_cnts)
    funstion_post_processing(image_file, sky_cnts, Fuzzy_cnts, Highlighted_cnts, img_ret, save_path)


if __name__ == '__main__':
    save_path = '../log_file'
    # image_base_path = r'../DataSet/True'
    image_base_path = r'Z:\extracted'
    model_path = r'..\model\u2net_bce_itr_640000_train_0.062078_tar_0.003078.pth'
    save_path = r'Z:\NailinLiao'
    _, image_list, _ = get_input_path(image_base_path)
    for image_file in image_list:
        try:
            print(image_file, '--------------------star')
            main(image_file, model_path, save_path)
        except:
            print(image_file, 'ERRO')
