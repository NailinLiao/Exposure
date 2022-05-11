import cv2


def seg_Highlighted_area(img_file_path, lowe=240, Debug=True):
    '''
     阈值法推理图像过曝
     :param image_rgb:rgb通道的array图像
     :param lowe:亮度检测阈值 推荐参数240《 X 《 254
     :param debug:调试按钮
     :return:是否过曝，过曝面积占比
     '''
    clea_rat = 0.05
    img = cv2.imread(img_file_path)
    img_ycr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_y = img_ycr[:, :, 0]
    h, w, _ = img_ycr.shape
    clea_area = h * w * clea_rat

    ret, thresh1 = cv2.threshold(img_y, lowe, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > clea_area:
            ret.append(cnt)
    if Debug:
        cv2.fillPoly(img, ret, (0, 0, 0))
        cv2.imshow('img', img)
        cv2.imshow('mask', thresh1)
        cv2.waitKey(0)
    return ret


if __name__ == '__main__':
    img_path = r'E:\WorkSpace\Exposure\DataSet\True\16409253741115660.jpeg'
    seg_Highlighted_area(img_path)
