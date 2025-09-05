import os
import shutil
import zipfile
import datetime
from tqdm import tqdm


def print_disk_usage(path="/kaggle/working"):
    """
    输出指定路径的磁盘总容量、已用容量和可用容量（单位GB）
    """
    usage = shutil.disk_usage(path)
    total_gb = usage.total / (1024 ** 3)
    used_gb = usage.used / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)

    print(f"磁盘路径: {path}")
    print(f"总容量: {total_gb:.2f} GB")
    print(f"已用: {used_gb:.2f} GB")
    print(f"可用: {free_gb:.2f} GB")


def delete_directory(path):
    """
    删除指定目录及以下所有文件
    path: 要删除的目录路径
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"目录已删除: {path}")
    else:
        print(f"目录不存在: {path}")


def zip_frames(packagePath, zipPath):
    """
    packagePath: 文件夹路径
    zipPath: 压缩包路径
    """
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirNames, fileNames in os.walk(packagePath):
        fpath = path.replace(packagePath, '')
        for name in fileNames:
            fullName = os.path.join(path, name)
            name = fpath + '\\' + name
            zip.write(fullName, name)
    zip.close()
    print("打包完成")
    print(datetime.datetime.utcnow())


def print_directory_structure(root_dir, indent="", directory_name="", is_root=True):
    """
    递归打印目录结构
    """
    if is_root:
        print(f"{directory_name}目录结构如下：\n")
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
            print_directory_structure(path, indent + extension, directory_name, is_root=False)


def process_loso_each(data_folder, loso_folder, num_classes, dataset_name="Dataset"):
    """
    每次只处理一个数据集的一种分类
    data_folder: 已分类的数据集 (例如 CASME2_retinaface_5)
    loso_folder: 输出路径 (例如 CASME2_retinaface_loso_5)
    num_classes: 分类数 (5 或 3 或 7 等)
    dataset_name: 用于 tqdm 的提示信息
    """

    # 提取所有被试前缀
    subjects = set()
    for class_folder in range(num_classes):
        class_path = os.path.join(data_folder, str(class_folder))
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            # CASMEⅡ SAMM  CAS(ME)^2 都是取文件名下划线分隔的第一个
            # CASMEⅡ sub01_EP19_05f_apex
            # SAMM 006_1_2_apex
            # CAS(ME)^2 spNO.1_a_355_apex
            if file.startswith("sub") or file.startswith("spNO"):
                subjects.add(file.split("_")[0])
            else:
                subjects.add(file.split("_")[0].zfill(3))
    subjects = sorted(subjects)

    # tqdm 进度条
    for subject in tqdm(subjects, desc=f"Processing {dataset_name} subjects"):
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
                    try:
                        shutil.copy(os.path.join(class_path, file), os.path.join(test_folder, file))
                    except OSError as e:
                        if e.errno == 28:  # 磁盘空间不足
                            print(f"[ERROR] 拷贝文件失败: {file}")
                            print_disk_usage("/kaggle/working")
                            raise
                        else:
                            raise
            # 训练集
            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                try:
                    shutil.copy(os.path.join(class_path, file), os.path.join(train_folder, file))
                except OSError as e:
                    if e.errno == 28:  # 磁盘空间不足
                        print(f"[ERROR] 拷贝文件失败: {file}")
                        print_disk_usage("/kaggle/working")
                        raise
                    else:
                        raise

def delete_main_2():
    casme2_dst_root_path = "/kaggle/working/CASME2_onset_apex_offset_retinaface"
    samm_dst_root_path = "/kaggle/working/SAMM_onset_apex_offset_retinaface"
    casme3_dst_root_path = "/kaggle/working/CASME3_onset_apex_offset_retinaface"
    casme2_output_folder = "/kaggle/working/CASME2_optflow_retinaface"
    samm_output_folder = "/kaggle/working/SAMM_optflow_retinaface"
    casme3_output_folder = "/kaggle/working/CASME3_optflow_retinaface"
    delete_directory(casme2_dst_root_path)
    delete_directory(samm_dst_root_path)
    delete_directory(casme3_dst_root_path)
    delete_directory(casme2_output_folder)
    delete_directory(samm_output_folder)
    delete_directory(casme3_output_folder)


if __name__ == "__main__":
    # 减少一些目录
    delete_main_2()
    # CASMEⅡ 数据集
    data_folder_5 = '/kaggle/working/CASME2_retinaface_5'  # 原始数据路径
    loso_folder_5 = '/kaggle/working/CASME2_retinaface_loso_5'  # 新路径
    data_folder_3 = '/kaggle/working/CASME2_retinaface_3'  # 原始数据路径
    loso_folder_3 = '/kaggle/working/CASME2_retinaface_loso_3'  # 新路径
    os.makedirs(loso_folder_5, exist_ok=True)
    os.makedirs(loso_folder_3, exist_ok=True)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_5, loso_folder_5, num_classes=5, dataset_name="CASMEⅡ")
    zipPath = '/kaggle/working/CASME2_retinaface_loso_5.zip'
    zip_frames(loso_folder_5, zipPath)
    print_directory_structure(loso_folder_5, directory_name='CASME2_retinaface_loso_5')
    # 减少一部分目录
    delete_directory(data_folder_5)
    delete_directory(loso_folder_5)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_3, loso_folder_3, num_classes=3, dataset_name="CASMEⅡ")
    zipPath = '/kaggle/working/CASME2_retinaface_loso_3.zip'
    zip_frames(loso_folder_3, zipPath)
    print_directory_structure(loso_folder_3, directory_name='CASME2_retinaface_loso_3')
    delete_directory(data_folder_3)
    delete_directory(loso_folder_3)

    # SAMM 数据集
    data_folder_3 = '/kaggle/working/SAMM_retinaface_3'  # 原始数据路径
    loso_folder_3 = '/kaggle/working/SAMM_retinaface_loso_3'  # 新路径
    os.makedirs(loso_folder_3, exist_ok=True)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_3, loso_folder_3, num_classes=3, dataset_name="SAMM")
    zipPath = '/kaggle/working/SAMM_retinaface_loso_3.zip'
    zip_frames(loso_folder_3, zipPath)
    print_directory_structure(loso_folder_3, directory_name='SAMM_retinaface_loso_3')
    delete_directory(data_folder_3)
    delete_directory(loso_folder_3)

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

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_7, loso_folder_7, num_classes=7, dataset_name="CAS(ME)^2")
    zipPath = '/kaggle/working/CASME3_retinaface_loso_7.zip'
    zip_frames(loso_folder_7, zipPath)
    print_directory_structure(loso_folder_7, directory_name='CASME3_retinaface_loso_7')
    delete_directory(data_folder_7)
    delete_directory(loso_folder_7)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_4, loso_folder_4, num_classes=4, dataset_name="CAS(ME)^2")
    zipPath = '/kaggle/working/CASME3_retinaface_loso_4.zip'
    zip_frames(loso_folder_4, zipPath)
    print_directory_structure(loso_folder_4, directory_name='CASME3_retinaface_loso_4')
    delete_directory(data_folder_4)
    delete_directory(loso_folder_4)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_3, loso_folder_3, num_classes=3, dataset_name="CAS(ME)^2")
    zipPath = '/kaggle/working/CASME3_retinaface_loso_3.zip'
    zip_frames(loso_folder_3, zipPath)
    print_directory_structure(loso_folder_3, directory_name='CASME3_retinaface_loso_3')
    delete_directory(data_folder_3)
    delete_directory(loso_folder_3)
