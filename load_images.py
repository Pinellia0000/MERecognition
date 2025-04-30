import glob
import cv2
from RetinaFace.tools import FaceDetector
import os


def crop_images_CASME2_retinaface():
    # face_det_model_path = "RetinaFace/Resnet50_Final.pth"
    # 这个pth文件有102MB 无法上传至github中
    # 使用的网络地址
    face_det_model_path = "/kaggle/input/retinaface-model/retinaface_Resnet50_Final.pth"
    face_detection = FaceDetector(face_det_model_path)

    # base_path = "Dataset/CASME2_onset_apex_offset_retinaface"
    # 原数据集
    base_path = "/kaggle/working/CASME2_RAW_selected"
    # # 被裁剪之后的数据集
    # copped_path = "/kaggle/working/casmeii/CASME2-RAW"


    # Iterate through all category folders
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):  # Make sure it's a directory
            for dir_crop_sub_vid_img in glob.glob(os.path.join(category_path, '*.jpg')):
                image = cv2.imread(dir_crop_sub_vid_img)

                h, w, c = image.shape

                face_left, face_top, face_right, face_bottom = \
                    face_detection.cal(image)

                img = image[face_top:face_bottom + 1,
                      face_left:face_right + 1, :]

                face = cv2.resize(img, (128, 128))  # Resize to 128x128

                cv2.imwrite(dir_crop_sub_vid_img, face)


if __name__ == '__main__':
    crop_images_CASME2_retinaface()
