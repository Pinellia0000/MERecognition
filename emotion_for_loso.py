import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile
import datetime


def zip_frames(packagePath, zipPath):
    """
    packagePath: 文件夹路径
    zipPath: 压缩包路径
    """
    zip = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirNames, fileNames in os.walk(packagePath):
        fpath = path.replace(packagePath, '')
        for name in fileNames:
            fullName = os.path.join(path, name)
            name = fpath + '\\' + name
            zip.write(fullName, name)
    zip.close()

def print_directory_structure(root_dir, indent=""):
    """
    递归打印目录结构
    """
    # 获取当前目录下的所有文件和文件夹，并排序（保证输出稳定）
    items = sorted(os.listdir(root_dir))

    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        # 判断是否是最后一个元素
        pointer = "└── " if idx == len(items) - 1 else "├── "
        print(indent + pointer + item)

        if os.path.isdir(path):
            # 如果是文件夹，递归打印子目录
            extension = "    " if idx == len(items) - 1 else "│   "
            print_directory_structure(path, indent + extension)


def main(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface, data_folder, annotation_file):
    # # 数据集路径
    # # 裁剪后的关键帧
    # CASME2_onset_apex_offset_retinaface = '/kaggle/working/CASME2_onset_apex_offset_retinaface'
    # # 光流图片
    # CASME2_optflow_retinaface = '/kaggle/working/CASME2_optflow_retinaface'
    # # 按情绪复制到对应文件夹
    # data_folder = '/kaggle/working/CASME2_retinaface'
    # # 注释文件
    # annotation_file = '/kaggle/input/casmeii/CASME2-coding-20140508.xlsx'

    # 情绪映射字典
    label_dict = {
        'happiness': 0,
        'surprise': 1,
        'disgust': 2,
        'repression': 3,
        'others': 4
    }
    # 以0、1、2、3、4作为文件夹名在data_folder下创建5个文件夹
    # 创建 0~4 文件夹
    for label in label_dict.values():
        os.makedirs(os.path.join(data_folder, str(label)), exist_ok=True)

    # 读取注释文件
    anno_df = pd.read_excel(annotation_file)

    # 遍历被试
    for sub_num in tqdm(range(1, 27), desc="Processing subjects"):
        sub_prefix = f'sub{sub_num:02d}'

        # 找出该被试的注释行
        sub_df = anno_df[anno_df['Subject'].apply(lambda x: f'sub{x:02d}') == sub_prefix]

        # 找出该被试的注释行
        # 根据第一层目录名称与第二层目录名称
        # 第一层目录名称的最后两位对应注释文件中的Subject
        # 第二层目录名称对应注释文件中的Filename
        # 根据这两个将对应行确定
        # 将获取该行的Estimated Emotion字段值
        # 根据获取的字段值将对应目录下的所有图片复制到字段字典对应的文件夹下
        # 未防止重复，复制的图片名称前增加原始第一层目录和第二年层目录名称，按下划线连接

        for _, row in sub_df.iterrows():
            filename = row['Filename']
            emotion = row['Estimated Emotion']

            # 获取情绪对应的类别编号
            if emotion not in label_dict:
                continue
            label_id = label_dict[emotion]

            # 源目录：关键帧和光流
            src_keyframe_dir = os.path.join(CASME2_onset_apex_offset_retinaface, sub_prefix, filename)
            src_optflow_dir = os.path.join(CASME2_optflow_retinaface, sub_prefix, filename)

            # 目标目录：根据类别编号放置
            dst_dir = os.path.join(data_folder, str(label_id))
            os.makedirs(dst_dir, exist_ok=True)

            # 复制关键帧
            if os.path.exists(src_keyframe_dir):
                for img_name in os.listdir(src_keyframe_dir):
                    src_path = os.path.join(src_keyframe_dir, img_name)
                    new_name = f"{sub_prefix}_{filename}_{img_name}"
                    dst_path = os.path.join(dst_dir, new_name)
                    shutil.copy(src_path, dst_path)

            # 复制光流图像
            if os.path.exists(src_optflow_dir):
                for img_name in os.listdir(src_optflow_dir):
                    src_path = os.path.join(src_optflow_dir, img_name)
                    new_name = f"{sub_prefix}_{filename}_{img_name}"
                    dst_path = os.path.join(dst_dir, new_name)
                    shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    # 数据集路径
    # 裁剪后的关键帧
    CASME2_onset_apex_offset_retinaface = '/kaggle/working/CASME2_onset_apex_offset_retinaface'
    # 光流图片
    CASME2_optflow_retinaface = '/kaggle/working/CASME2_optflow_retinaface'
    # 按情绪复制到对应文件夹
    data_folder = '/kaggle/working/CASME2_retinaface'
    # 注释文件
    annotation_file = '/kaggle/input/casmeii/CASME2-coding-20140508.xlsx'
    main(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface, data_folder, annotation_file)
    zipPath = '/kaggle/working/CASME2_retinaface.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("目录结构如下：\n")
    print_directory_structure(data_folder)
