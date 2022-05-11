import cv2


class ReadImage:
    def __init__(self):
        pass

    @staticmethod
    def read_img(image_file_path):
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def read_img_by_centre(image_file_path, rat=0.5):
        image = cv2.imread(image_file_path)
        h, w, _ = image.shape
        h_rat, w_rat = int(h * rat), int(w * rat)
        cut_h = int((h - h_rat) / 2)
        cut_w = int((w - w_rat) / 2)
        image = cv2.cvtColor(image[cut_h:-cut_h, cut_w:-cut_w], cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def read_img_by_centre_donw(image_file_path, rat=0.5):
        image = cv2.imread(image_file_path)
        h, w, _ = image.shape
        h_rat, w_rat = int(h * rat), int(w * rat)
        cut_w = int((w - w_rat) / 2)
        image = cv2.cvtColor(image[h - h_rat:, cut_w:-cut_w], cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def read_img_by_master_area(image_file_path, h_rat=0.5, w_rat=0.3):
        image = cv2.imread(image_file_path)
        h, w, _ = image.shape
        h_rat, w_rat = int(h * h_rat), int(w * w_rat)

        image = cv2.cvtColor(image[h - h_rat:, w_rat:-w_rat], cv2.COLOR_BGR2RGB)

        return image
