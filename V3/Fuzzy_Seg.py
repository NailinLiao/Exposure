import cv2
import numpy as np


def calLaplacianVar(img: np.array):
    sobelImg = cv2.Laplacian(img, cv2.CV_64FC1)
    # 标准差
    mu, sigma = cv2.meanStdDev(sobelImg)
    sigmaValue = sigma[0][0]
    # 方差
    variance = pow(sigmaValue, 2)
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=5)
    dst = cv2.convertScaleAbs(gray_lap)
    GRAY = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(GRAY, 20, 255, cv2.THRESH_BINARY_INV)
    cnts, mask = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 500:
            new_cnts.append(cnt)
    # cv2.imshow('thresh1', thresh1)
    # cv2.waitKey(0)
    return thresh1, new_cnts


def seg_Fuzzy(img_file_path, Debug=True):
    clea_rat = 0.05
    img = cv2.imread(img_file_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    clea_area = h * w * clea_rat
    thresh1, _ = calLaplacianVar(img)
    k = np.ones((3, 3), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, k)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, k, iterations=15)

    cnts, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ret = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > clea_area:
            ret.append(cnt)
    if Debug:
        mask = np.zeros_like(thresh1)
        mask = cv2.fillPoly(mask, ret, (255, 0, 0))

        cv2.imshow('mask', mask)
        cv2.imshow('thresh1', thresh1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return ret


if __name__ == '__main__':
    img_path = r'E:\WorkSpace\Exposure\DataSet\True\16409253741115660.jpeg'
    seg_Fuzzy(img_path)
