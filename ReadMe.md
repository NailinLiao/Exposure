# 曝光检测

---

### 检测方式

- YCR 明度阈值检测
- 参考V1.main.py

`` 单张检测，并生成 检测结果日志

    '''
    def one_check(img_file_path, log_save_path=None):
    检查单张图像
    :param img_file_path:待检测图像的路径，XXX/xxx/xxx.png
    :param log_save_path:检测日志保存位置，若不指定，则自动存入用户跟根目录下 ~/DeepWay/DeepWay/overexposure_log/xxx_overexposure_Check_Log.csv
    :return:bool 0 为正常 1 为过曝
    '''
    statu = exposure_file(img_file_path)
    wirte_Check_log_file(img_file_path, statu, log_save_path)
    print(statu)
    if statu != 0:
        print('图像:', img_file_path, '  检测结果为   过曝')
    else:
        print('图像:', img_file_path, '  检测结果为   正常')
    return statu

``

`` 批量检测，并生成 检测结果日志

    '''
    def batch_check(image_base_path, log_save_path=None):
    批量检测图像，并写入日志
    :param image_base_path:待检测的图片根路径
    :param log_save_path:检测日志保存位置，若不指定，则自动存入用户跟根目录下 ~/DeepWay/DeepWay/overexposure_log/xxx_overexposure_Check_Log.csv
    :return:None
    '''
    _, image_list, _ = get_input_path(image_base_path)
    for image_file in image_list:
        statu = exposure_file(image_file)
        wirte_Check_log_file(image_file, statu, log_save_path)
        print(statu)

        if statu != 0:
            print('图像:', image_file, '  检测结果为   过曝')
        else:
            print('图像:', image_file, '  检测结果为   正常')

``

| 检测方法 | 准确率   | 精度   |召回率    |
|------|-------|------|-------|
| 阈值   | 83.5% | 100% |67%|
| 模型   | 86.0% | 86%  |96%|

---