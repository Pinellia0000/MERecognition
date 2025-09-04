import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile
import datetime


def zip_frames(packagePath, zipPath):
    """将目录打包成 zip 文件"""
    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path, _, fileNames in os.walk(packagePath):
            fpath = os.path.relpath(path, packagePath)
            for name in fileNames:
                fullName = os.path.join(path, name)
                arcname = os.path.join(fpath, name) if fpath != "." else name
                zipf.write(fullName, arcname)


def print_directory_structure(root_dir, indent=""):
    """递归打印目录结构"""
    if not os.path.exists(root_dir):
        print(f"目录不存在: {root_dir}")
        return

    items = sorted(os.listdir(root_dir))
    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        pointer = "└── " if idx == len(items) - 1 else "├── "
        print(indent + pointer + item)

        if os.path.isdir(path):
            extension = "    " if idx == len(items) - 1 else "│   "
            print_directory_structure(path, indent + extension)


def CASME2_5c_3c(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface,
                 data_folder_5, data_folder_3, annotation_file):
    """
    关键帧结构
    ├── sub01
    │   ├── sub01_EP02_01f_apex.jpg
    │   ├── sub01_EP02_01f_offset.jpg
    │   ├── sub01_EP02_01f_onset.jpg
    │   ├── sub01_EP03_02_apex.jpg
    光流帧结构
    ├── sub01
    │   ├── sub01_EP02_01f_1_u.jpg
    │   ├── sub01_EP02_01f_1_v.jpg
    │   ├── sub01_EP02_01f_2_u.jpg
    │   ├── sub01_EP02_01f_2_v.jpg

    CASME2 5分类 + 3分类 整理
    sadness fear 数量太少 不参与分类
    5类：'happiness':0, 'surprise':1, 'disgust':2, 'repression':3, 'others':4
    3类：'positive':0, 'negative':1, 'surprise':2
         positive = happiness
         negative = disgust + repression
         surprise = surprise
         others 不参与 3分类
    """

    # 5分类字典
    label_dict_5 = {
        'happiness': 0,
        'surprise': 1,
        'disgust': 2,
        'repression': 3,
        'others': 4
    }

    # 3分类字典（没有 others）
    label_dict_3 = {
        'happiness': 0,  # positive
        'disgust': 1,  # negative
        'repression': 1,  # negative
        'surprise': 2  # surprise
    }

    # 创建 5分类目录
    for label in sorted(set(label_dict_5.values())):
        os.makedirs(os.path.join(data_folder_5, str(label)), exist_ok=True)

    # 创建 3分类目录（不包含 others）
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    # 读取注释文件
    anno_df = pd.read_excel(annotation_file)
    # 找出该被试的注释行
    # 第一层目录名称的最后两位对应注释文件中的Subject
    # 图片名称中如果包含对应注释文件中的Filename
    # 根据这两个将对应行确定
    # 将获取该行的Estimated Emotion字段值
    # 根据获取的字段值将对应目录下的所有图片复制到字段字典对应的文件夹下
    # 遍历被试
    # 遍历 sub01 ~ sub26
    for sub_num in tqdm(range(1, 27), desc="Processing subjects"):
        sub_prefix = f'sub{sub_num:02d}'
        sub_folder_path = os.path.join(CASME2_onset_apex_offset_retinaface, sub_prefix)
        if not os.path.exists(sub_folder_path):
            continue

        # 筛选该被试的注释行
        sub_df = anno_df[anno_df['Subject'].apply(lambda x: f'sub{x:02d}') == sub_prefix]

        def process_and_copy(src_folder):
            if not os.path.exists(src_folder):
                return
            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)

                matched_rows = sub_df[sub_df['Filename'].apply(lambda x: img_name.startswith(str(x)))]
                if matched_rows.empty:
                    continue

                for _, row in matched_rows.iterrows():
                    emotion = row['Estimated Emotion']
                    if emotion not in label_dict_5:
                        continue

                    # ---- 5分类 ----
                    label_id_5 = label_dict_5[emotion]
                    dst_dir_5 = os.path.join(data_folder_5, str(label_id_5))
                    new_name = f"{sub_prefix}_{img_name}"
                    dst_path_5 = os.path.join(dst_dir_5, new_name)
                    if not os.path.exists(dst_path_5):
                        shutil.copy(img_path, dst_path_5)

                    # ---- 3分类 ---- (others 不参与)
                    if emotion in label_dict_3:
                        label_id_3 = label_dict_3[emotion]
                        dst_dir_3 = os.path.join(data_folder_3, str(label_id_3))
                        dst_path_3 = os.path.join(dst_dir_3, new_name)
                        if not os.path.exists(dst_path_3):
                            shutil.copy(img_path, dst_path_3)

        # 关键帧和光流都处理
        process_and_copy(sub_folder_path)
        process_and_copy(os.path.join(CASME2_optflow_retinaface, sub_prefix))


def SAMM_3c(SAMM_onset_apex_offset_retinaface, SAMM_optflow_retinaface,
            data_folder_3, annotation_file):
    """
    关键帧结构
    ├── 006
    │   ├── 006_1_2_apex.jpg
    │   ├── 006_1_2_offset.jpg
    │   ├── 006_1_2_onset.jpg
    │   ├── 006_1_3_apex.jpg
    光流帧结构
    ├── 006
    │   ├── 006_1_2_1_u.jpg
    │   ├── 006_1_2_1_v.jpg
    │   ├── 006_1_2_2_u.jpg
    │   ├── 006_1_2_2_v.jpg
    Other 不参与 3分类
    SAMM 3分类 整理
    3类：'positive':0, 'negative':1, 'surprise':2
         positive = Happiness
         negative = Anger + Disgust +  Contempt + Sadness + Fear
         surprise = Surprise
    """
    # 注意：# samm 数据不是从第一行开始，前几行有说明性文字
    #     df = pd.read_excel(excel_path, header=13)  # 列名在第14行
    # 找出该被试的注释行
    # 第一层目录名称为对应注释文件中的Subject
    # 图片名称中如果包含对应注释文件中的Filename
    # 根据这两个将对应行确定
    # 将获取该行的Estimated Emotion字段值
    # 根据获取的字段值将对应目录下的所有图片复制到字段字典对应的文件夹下
    # 遍历被试
    label_dict_3 = {
        'Happiness': 0, 'Anger': 1, 'Disgust': 1, 'Contempt': 1, 'Sadness': 1, 'Fear': 1, 'Surprise': 2
    }
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    anno_df = pd.read_excel(annotation_file, header=13)

    for subject in tqdm(os.listdir(SAMM_onset_apex_offset_retinaface), desc="Processing SAMM"):
        sub_folder_path = os.path.join(SAMM_onset_apex_offset_retinaface, subject)
        sub_df = anno_df[anno_df['Subject'] == int(subject)]

        def process_and_copy(src_folder):
            if not os.path.exists(src_folder):
                return
            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)
                matched_rows = sub_df[sub_df['Filename'].apply(lambda x: str(x) in img_name)]
                if matched_rows.empty:
                    continue
                for _, row in matched_rows.iterrows():
                    emotion = row['Emotion']
                    if emotion not in label_dict_3:
                        continue
                    label_id = label_dict_3[emotion]
                    dst_dir = os.path.join(data_folder_3, str(label_id))
                    shutil.copy(img_path, os.path.join(dst_dir, f"{subject}_{img_name}"))

        process_and_copy(sub_folder_path)
        process_and_copy(os.path.join(SAMM_optflow_retinaface, subject))


def CASME3_7c_4c_3c(CASME3_onset_apex_offset_retinaface, CASME3_optflow_retinaface,
                    data_folder_7, data_folder_4, data_folder_3, annotation_file):
    """
    关键帧结构
    ├── spNO.1
    │   ├── spNO.1_a_355_apex.jpg
    │   ├── spNO.1_a_355_offset.jpg
    │   ├── spNO.1_a_355_onset.jpg
    │   ├── spNO.1_b_166_apex.jpg
    │   ├── spNO.1_b_166_offset.jpg
    光流帧结构
    ├── spNO.1
    │   ├── spNO.1_a_355_1_u.jpg
    │   ├── spNO.1_a_355_1_v.jpg
    │   ├── spNO.1_a_355_2_u.jpg
    │   ├── spNO.1_a_355_2_v.jpg
    CASME3 7分类 4分类 3分类
    7类：'happy':0, 'surprise':1, 'disgust':2, 'anger':3, 'fear':4, 'sad':5, 'others':6
    SAMM 3分类 整理  others 不参与 3分类
    3类：'positive':0, 'negative':1, 'surprise':2
         positive = happy
         negative = disgust + anger +  fear + sad
         surprise = surprise
    SAMM 3分类 整理  others 参与 4分类
    4类：'positive':0, 'negative':1, 'surprise':2, 'others':3
         positive = happy
         negative = disgust + anger +  fear + sad
         surprise = surprise
         others = others
    """
    # 找出该被试的注释行
    # 第一层目录名称为对应注释文件中的Subject
    # 图片名称中如果包含对应注释文件中的Filename
    # 根据这两个将对应行确定
    # 将获取该行的emotion字段值
    # 根据获取的字段值将对应目录下的所有图片复制到字段字典对应的文件夹下
    # 遍历被试
    label_dict_7 = {'happy': 0, 'surprise': 1, 'disgust': 2, 'anger': 3, 'fear': 4, 'sad': 5, 'others': 6}
    label_dict_3 = {'happy': 0, 'disgust': 1, 'anger': 1, 'fear': 1, 'sad': 1, 'surprise': 2}
    label_dict_4 = {'happy': 0, 'disgust': 1, 'anger': 1, 'fear': 1, 'sad': 1, 'surprise': 2, 'others': 3}

    for label in sorted(set(label_dict_7.values())):
        os.makedirs(os.path.join(data_folder_7, str(label)), exist_ok=True)
    for label in sorted(set(label_dict_4.values())):
        os.makedirs(os.path.join(data_folder_4, str(label)), exist_ok=True)
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    anno_df = pd.read_excel(annotation_file)

    for subject in tqdm(os.listdir(CASME3_onset_apex_offset_retinaface), desc="Processing CASME3"):
        sub_folder_path = os.path.join(CASME3_onset_apex_offset_retinaface, subject)
        sub_df = anno_df[anno_df['Subject'] == subject]

        def process_and_copy(src_folder):
            if not os.path.exists(src_folder):
                return
            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)
                matched_rows = sub_df[sub_df['Filename'].apply(lambda x: str(x) in img_name)]
                if matched_rows.empty:
                    continue
                for _, row in matched_rows.iterrows():
                    emotion = str(row['emotion']).lower()
                    if emotion not in label_dict_7:
                        continue
                    new_name = f"{subject}_{img_name}"

                    shutil.copy(img_path, os.path.join(data_folder_7, str(label_dict_7[emotion]), new_name))
                    if emotion in label_dict_4:
                        shutil.copy(img_path, os.path.join(data_folder_4, str(label_dict_4[emotion]), new_name))
                    if emotion in label_dict_3:
                        shutil.copy(img_path, os.path.join(data_folder_3, str(label_dict_3[emotion]), new_name))

        process_and_copy(sub_folder_path)
        process_and_copy(os.path.join(CASME3_optflow_retinaface, subject))


if __name__ == '__main__':
    # CASMEⅡ
    # 数据集路径
    CASME2_onset_apex_offset_retinaface = '/kaggle/working/CASME2_onset_apex_offset_retinaface'
    CASME2_optflow_retinaface = '/kaggle/working/CASME2_optflow_retinaface'
    data_folder_5 = '/kaggle/working/CASME2_retinaface_5'
    data_folder_3 = '/kaggle/working/CASME2_retinaface_3'
    annotation_file = '/kaggle/input/casmeii/CASME2-coding-20140508.xlsx'

    # 整理数据
    CASME2_5c_3c(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface,
                 data_folder_5, data_folder_3, annotation_file)

    # 打包 5分类
    zipPath = '/kaggle/working/CASME2_retinaface_5.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_5, zipPath)
    print("5分类打包完成")
    print_directory_structure(data_folder_5)

    # 打包 3分类
    zipPath = '/kaggle/working/CASME2_retinaface_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_3, zipPath)
    print("3分类打包完成")
    print_directory_structure(data_folder_3)

    print("全部完成")
    print(datetime.datetime.utcnow())

    # SAMM
    # 数据集路径
    SAMM_onset_apex_offset_retinaface = '/kaggle/working/SAMM_onset_apex_offset_retinaface'
    SAMM_optflow_retinaface = '/kaggle/working/SAMM_optflow_retinaface'
    data_folder_3 = '/kaggle/working/SAMM_retinaface_3'
    annotation_file = '/kaggle/input/samm-dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'

    # 整理数据
    SAMM_3c(SAMM_onset_apex_offset_retinaface, SAMM_optflow_retinaface,
            data_folder_3, annotation_file)

    # 打包 5分类
    zipPath = '/kaggle/working/SAMM_retinaface_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_3, zipPath)
    print("3分类打包完成")
    print_directory_structure(data_folder_3)

    # CASME3
    # 数据集路径
    CASME3_onset_apex_offset_retinaface = '/kaggle/working/CASME3_onset_apex_offset_retinaface'
    CASME3_optflow_retinaface = '/kaggle/working/CASME3_optflow_retinaface'
    data_folder_7 = '/kaggle/working/CASME3_retinaface_7'
    data_folder_4 = '/kaggle/working/CASME3_retinaface_4'
    data_folder_3 = '/kaggle/working/CASME3_retinaface_3'
    annotation_file = '/kaggle/input/casme3/cas(me)3_part_A_ME_label_JpgIndex_v2_20250903.xlsx'

    # 整理数据
    CASME3_7c_4c_3c(CASME3_onset_apex_offset_retinaface, CASME3_optflow_retinaface,
                 data_folder_7, data_folder_4, data_folder_3, annotation_file)
    # 打包 7分类
    zipPath = '/kaggle/working/CASME2_retinaface_7.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_7, zipPath)
    print("5分类打包完成")
    print_directory_structure(data_folder_7)

    # 打包 4分类
    zipPath = '/kaggle/working/CASME3_retinaface_4.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_4, zipPath)
    print("5分类打包完成")
    print_directory_structure(data_folder_4)

    # 打包 3分类
    zipPath = '/kaggle/working/CASME3_retinaface_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_3, zipPath)
    print("3分类打包完成")
    print_directory_structure(data_folder_3)

    print("全部完成")
    print(datetime.datetime.utcnow())
