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


def main(data_folder):
    # data_folder 只包含关键帧和光流图像
    # 在LOSO划分后 直接用于训练
    # # data_folder = 'Dataset/CASME2_retinaface'
    # data_folder = '/kaggle/working/CASME2_retinaface'

    for sub_num in range(1, 27):
        sub_prefix = f'sub{sub_num:02d}'
        sub_folder = os.path.join(data_folder, sub_prefix)
        os.makedirs(sub_folder, exist_ok=True)

        for class_folder in range(5):
            class_path = os.path.join(data_folder, str(class_folder))

            files = [file for file in os.listdir(class_path) if file.startswith(sub_prefix)]

            not_files = [file for file in os.listdir(class_path) if not file.startswith(sub_prefix)]

            if len(files) == 0:
                pass
            else:
                test_folder = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_folder, exist_ok=True)
                for file in files:
                    shutil.copy(os.path.join(class_path, file), os.path.join(test_folder, file))

            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_folder, file))


if __name__ == "__main__":
    data_folder = '/kaggle/working/CASME2_retinaface'
    main(data_folder)
    zipPath = '/kaggle/working/CASME2_retinaface_loso.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("目录结构如下：\n")
    print_directory_structure(data_folder)
