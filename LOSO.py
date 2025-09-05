import os
import shutil
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


def process_loso(data_folder, loso_folder, num_classes):
    """
    data_folder: 已分类的数据集 (例如 CASME2_retinaface_5)
    loso_folder: 输出路径 (例如 CASME2_retinaface_loso_5)
    num_classes: 分类数 (5 或 3)
    """

    # 提取所有被试前缀
    subjects = set()
    for class_folder in range(num_classes):
        class_path = os.path.join(data_folder, str(class_folder))
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            if file.startswith("sub"):
                subjects.add(file.split("_")[0])
    subjects = sorted(subjects)

    # 遍历被试
    for subject in subjects:
        sub_folder = os.path.join(loso_folder, subject)
        os.makedirs(sub_folder, exist_ok=True)

        for class_folder in range(num_classes):
            class_path = os.path.join(data_folder, str(class_folder))
            if not os.path.exists(class_path):
                continue

            files = [file for file in os.listdir(class_path) if file.startswith(subject)]
            not_files = [file for file in os.listdir(class_path) if not file.startswith(subject)]

            # 测试集
            if files:
                test_folder = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_folder, exist_ok=True)
                for file in files:
                    shutil.copy(os.path.join(class_path, file), os.path.join(test_folder, file))

            # 训练集
            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_folder, file))


if __name__ == "__main__":
    # CASMEⅡ 数据集
    data_folder_5 = '/kaggle/working/CASME2_retinaface_5'  # 原始数据路径
    loso_folder_5 = '/kaggle/working/CASME2_retinaface_loso_5'  # 新路径
    data_folder_3 = '/kaggle/working/CASME2_retinaface_3'  # 原始数据路径
    loso_folder_3 = '/kaggle/working/CASME2_retinaface_loso_3'  # 新路径
    os.makedirs(loso_folder_5, exist_ok=True)
    os.makedirs(loso_folder_3, exist_ok=True)

    process_loso(data_folder_5, loso_folder_5, num_classes=5)
    zipPath = '/kaggle/working/CASME2_retinaface_loso_5.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(loso_folder_5, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASME2 5分类 目录结构如下：\n")
    print_directory_structure(loso_folder_5)

    process_loso(data_folder_3, loso_folder_3, num_classes=3)
    zipPath = '/kaggle/working/CASME2_retinaface_loso_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(loso_folder_3, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASME2 3分类 目录结构如下：\n")
    print_directory_structure(loso_folder_3)

    # SAMM 数据集
    data_folder_3 = '/kaggle/working/SAMM_retinaface_3'  # 原始数据路径
    loso_folder_3 = '/kaggle/working/SAMM_retinaface_loso_3'  # 新路径
    os.makedirs(loso_folder_3, exist_ok=True)

    process_loso(data_folder_3, loso_folder_3, num_classes=3)
    zipPath = '/kaggle/working/SAMM_retinaface_loso_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(loso_folder_3, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("SAMM 3分类 目录结构如下：\n")
    print_directory_structure(loso_folder_3)

    # CAS(ME)^2 数据集
    data_folder_7 = '/kaggle/working/CASME3_retinaface_7'  # 原始数据路径
    loso_folder_7 = '/kaggle/working/CASME3_retinaface_loso_7'  # 新路径
    data_folder_4 = '/kaggle/working/CASME3_retinaface_4'  # 原始数据路径
    loso_folder_4 = '/kaggle/working/CASME3_retinaface_loso_4'  # 新路径
    data_folder_3 = '/kaggle/working/CASME3_retinaface_3'  # 原始数据路径
    loso_folder_3 = '/kaggle/working/CASME3_retinaface_loso_3'  # 新路径
    os.makedirs(loso_folder_7, exist_ok=True)
    os.makedirs(loso_folder_4, exist_ok=True)
    os.makedirs(loso_folder_3, exist_ok=True)

    process_loso(data_folder_7, loso_folder_7, num_classes=7)
    zipPath = '/kaggle/working/CASME3_retinaface_loso_7.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(loso_folder_7, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASME3 7分类 目录结构如下：\n")
    print_directory_structure(loso_folder_7)

    process_loso(data_folder_4, loso_folder_4, num_classes=4)
    zipPath = '/kaggle/working/CASME3_retinaface_loso_4.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(loso_folder_4, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASME3 4分类 目录结构如下：\n")
    print_directory_structure(loso_folder_4)

    process_loso(data_folder_3, loso_folder_3, num_classes=3)
    zipPath = '/kaggle/working/CASME3_retinaface_loso_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(loso_folder_3, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASME3 3分类 目录结构如下：\n")
    print_directory_structure(loso_folder_3)
