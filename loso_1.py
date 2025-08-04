import os
import shutil
from tqdm import tqdm  # 添加进度条
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


def organize_casme2_by_subject(input_folder, output_folder):
    """
    将按类别分类的 CASME2 数据重新组织为按受试者（Subject）分类，并划分训练集/测试集。

    参数:
        input_folder (str): 输入文件夹路径（包含类别子文件夹，如 `0/`, `1/`, ..., `4/`）。
        output_folder (str): 输出文件夹路径（按 `subXX/train/class/` 和 `subXX/test/class/` 组织）。
    """
    # 遍历所有受试者（sub01 到 sub26）
    for sub_num in tqdm(range(1, 27), desc="组织受试者数据"):
        sub_prefix = f'sub{sub_num:02d}'  # 格式化为 sub01, sub02, ..., sub26
        sub_folder = os.path.join(output_folder, sub_prefix)
        os.makedirs(sub_folder, exist_ok=True)

        # 遍历所有类别（0 到 4）
        for class_folder in range(5):
            class_path = os.path.join(input_folder, str(class_folder))

            # 获取当前受试者的文件（作为测试集）
            test_files = [f for f in os.listdir(class_path) if f.startswith(sub_prefix)]
            # 其他文件作为训练集
            train_files = [f for f in os.listdir(class_path) if not f.startswith(sub_prefix)]

            # 处理测试集
            if test_files:
                test_dst = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_dst, exist_ok=True)
                for file in test_files:
                    src = os.path.join(class_path, file)
                    dst = os.path.join(test_dst, file)
                    shutil.copy(src, dst)

            # 处理训练集
            if train_files:
                train_dst = os.path.join(sub_folder, 'train', str(class_folder))
                os.makedirs(train_dst, exist_ok=True)
                for file in train_files:
                    src = os.path.join(class_path, file)
                    dst = os.path.join(train_dst, file)
                    shutil.copy(src, dst)


if __name__ == "__main__":
    # 输入：光流特征文件夹（来自之前的脚本）
    input_folder = "/kaggle/working/CASME2_optflow_retinaface"
    # 输出：按受试者组织的文件夹
    output_folder = "/kaggle/working/CASME2_organized_by_subject"
    organize_casme2_by_subject(input_folder, output_folder)
    print("数据组织完成！")
    zipPath = '/kaggle/working/CASME2_organized_by_subject.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(output_folder, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
