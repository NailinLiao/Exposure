'''
脚本内容:
    实现批量\单张过曝检测
        - 并读写检测日志
    更新:
        V2.01 增加 对天空的 分割模型 ,将天空进行掩盖 使用阈值判断过曝
            时间:20220325


'''
from Exposure_Check import *
import shutil
from log.log_write import *
from Sky_Seg import *


def exposure_file(img_file, img, save_path=None, rat=0.5):
    '''
    按照区域区分，切依次检查，则可以得到准确的过曝区域
        中心偏下存在过曝
        中心存在过曝
        左右及上部边缘存在过曝
    :return:
    '''
    if save_path != None:
        All_overproof_path = os.path.join(save_path, 'All_overproof')
        Center_overproof_path = os.path.join(save_path, 'Center_overproof')
        Down_overproof_path = os.path.join(save_path, 'Down_overproof')
        No_exposure = os.path.join(save_path, 'No_exposure')
        fileName = os.path.split(img_file)[-1]
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(All_overproof_path)
            os.makedirs(Center_overproof_path)
            os.makedirs(Down_overproof_path)
            os.makedirs(No_exposure)

    state, degree = ExposureCheck().ThresholdInference(img, debug=False)
    if state:
        if save_path != None:
            end_path = os.path.join(Down_overproof_path, fileName)
            shutil.copyfile(img_file, end_path)
        return 1

    img = ReadImage.read_img_by_centre(img_file, rat=rat)
    state, degree = ExposureCheck().ThresholdInference(img, debug=False)
    if state:
        if save_path != None:
            end_path = os.path.join(Center_overproof_path, fileName)
            shutil.copyfile(img_file, end_path)
        return 1

    img = ReadImage.read_img(img_file)
    state, degree = ExposureCheck().ThresholdInference(img, debug=False)
    if state:
        if save_path != None:
            end_path = os.path.join(All_overproof_path, fileName)
            shutil.copyfile(img_file, end_path)
        return 1
    if save_path != None:
        end_path = os.path.join(No_exposure, fileName)
        shutil.copyfile(img_file, end_path)
    return 0


def batch_check(image_base_path, model_path, log_save_path=None):
    '''
    批量检测图像，并写入日志
    :param image_base_path:待检测的图片根路径
    :param log_save_path:检测日志保存位置，若不指定，则自动存入用户跟根目录下 ~/DeepWay/DeepWay/overexposure_log/xxx_overexposure_Check_Log.csv
    :return:None
    '''
    skySeg = SkySeg(model_path)

    _, image_list, _ = get_input_path(image_base_path)
    for image_file in image_list:
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = 255 - skySeg.inference(img)
        img = cv2.bitwise_and(img, out)
        statu = exposure_file(image_file, img)
        wirte_Check_log_file(image_file, statu, log_save_path)
        if statu != 0:
            print('图像:', image_file, '  检测结果为   过曝')
        else:
            print('图像:', image_file, '  检测结果为   正常')


def one_check(img_file_path, log_save_path=None):
    '''
    检查单张图像
    :param img_file_path:待检测图像的路径，XXX/xxx/xxx.png
    :param log_save_path:检测日志保存位置，若不指定，则自动存入用户跟根目录下 ~/DeepWay/DeepWay/overexposure_log/xxx_overexposure_Check_Log.csv
    :return:bool 0 为正常 1 为过曝
    '''
    skySeg = SkySeg(model_path)
    img = cv2.imread(img_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = 255 - skySeg.inference(img)
    img = cv2.bitwise_and(img, out)
    statu = exposure_file(img_file_path, img)
    wirte_Check_log_file(img_file_path, statu, log_save_path)
    print(statu)
    if statu != 0:
        print('图像:', img_file_path, '  检测结果为   过曝')
    else:
        print('图像:', img_file_path, '  检测结果为   正常')
    return statu


if __name__ == '__main__':
    log_path = '../log_file'
    image_base_path = r'../DataSet/True'
    model_path = r'E:\WorkSpace\Exposure\model\u2net_bce_itr_640000_train_0.062078_tar_0.003078.pth'
    # 批量预测
    batch_check(image_base_path, model_path, log_save_path=log_path)

    # 单站预测
    # img_file_path = r'../DataSet/Image/normal/000001.png'
    # one_check(img_file_path, log_save_path=log_path)
