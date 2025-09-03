import os
import shutil
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库
import zipfile
import datetime
import glob


# 安全解析整数
def safe_parse_int(value):
    """
    要求起始帧、顶点帧和结束帧的图片编号在注释文件中存在
    只要有一项不存在 就跳过该样本的处理
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


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


def get_casme2_onset_apex_offset(src_root, dst_root, excel_path):
    """
    从 CASME2 数据集中提取起始帧、顶点帧和结束帧，并将其保存到目标目录
    保存结构：subXX/EPxx_xxf_onset.jpg, apex.jpg, offset.jpg
    """
    os.makedirs(dst_root, exist_ok=True)

    df = pd.read_excel(excel_path)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        subject = str(row['Subject']).strip()  # e.g., '01'
        filename = str(row['Filename']).strip()  # e.g., 'EP02_01f'
        onset = safe_parse_int(row['OnsetFrame'])
        apex = safe_parse_int(row['ApexFrame'])
        offset = safe_parse_int(row['OffsetFrame'])

        if None in (onset, apex, offset):
            print(f"[SKIP] Invalid frame data for: {filename}")
            continue

        video_folder = os.path.join(src_root, f"sub{subject.zfill(2)}", filename)
        dst_folder = os.path.join(dst_root, f"sub{subject.zfill(2)}")
        os.makedirs(dst_folder, exist_ok=True)

        for frame_type, frame_id in [('onset', onset), ('apex', apex), ('offset', offset)]:
            img_name = f"img{frame_id}.jpg"
            src_img_path = os.path.join(video_folder, img_name)

            if os.path.exists(src_img_path):
                # 新的命名规则：视频名 + "_" + 帧类型
                dst_img_name = f"{filename}_{frame_type}.jpg"
                dst_img_path = os.path.join(dst_folder, dst_img_name)

                shutil.copy(src_img_path, dst_img_path)
                print(f"Copied: {src_img_path} → {dst_img_path}")
            else:
                print(f"[WARNING] Not found: {src_img_path}")


def get_samm_onset_apex_offset(src_root, dst_root, excel_path):
    """
    从 SAMM 数据集中提取起始帧、顶点帧和结束帧，并将其保存到目标目录
    保存结构：006/006_1_2_onset.jpg, 006_1_2_apex.jpg, 006_1_2_offset.jpg
    """
    os.makedirs(dst_root, exist_ok=True)

    # 手动或自动确定 header 行
    # samm数据不是从第一行开始 前几行有说明性文字
    df = pd.read_excel(excel_path, header=13)  # 列名在第14行

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing SAMM videos"):
        subject = str(row['Subject']).zfill(3)  # e.g., '006'
        filename = str(row['Filename']).strip()  # e.g., '006_1_2'

        onset = safe_parse_int(row['Onset Frame'])
        apex = safe_parse_int(row['Apex Frame'])
        offset = safe_parse_int(row['Offset Frame'])

        if None in (onset, apex, offset):
            print(f"[SKIP] Invalid frame data for: {filename}")
            continue

        video_folder = os.path.join(src_root, subject, filename)
        dst_folder = os.path.join(dst_root, subject)
        os.makedirs(dst_folder, exist_ok=True)

        for frame_type, frame_id in [('onset', onset), ('apex', apex), ('offset', offset)]:
            # 模糊匹配：既能匹配 4 位，也能匹配 5 位
            pattern = os.path.join(video_folder, f"{subject}_{frame_id:04d}*.jpg")
            candidates = glob.glob(pattern)

            if len(candidates) > 0:
                src_img_path = candidates[0]  # 取第一个匹配的
                dst_img_name = f"{filename}_{frame_type}.jpg"
                dst_img_path = os.path.join(dst_folder, dst_img_name)

                shutil.copy(src_img_path, dst_img_path)
                print(f"Copied: {src_img_path} → {dst_img_path}")
            else:
                print(f"[WARNING] Not found: {subject}_{frame_id} in {video_folder}")



if __name__ == "__main__":
    # CASME2 数据集
    # 路径配置
    casme2_src_root = '/kaggle/input/casmeii/CASME2-RAW/CASME2-RAW'
    casme2_dst_root = '/kaggle/working/CASME2_onset_apex_offset'
    # 读取 Excel 标注文件
    # sub04 EP12_01f 的顶点帧在注释文件中没有给出 标记为/
    casme2_excel_path = '/kaggle/input/casmeii/CASME2-coding-20140508.xlsx'
    get_casme2_onset_apex_offset(casme2_src_root, casme2_dst_root, casme2_excel_path)
    zipPath = '/kaggle/working/CASME2_onset_apex_offset.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(casme2_dst_root, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASME2关键帧目录结构如下：\n")
    print_directory_structure(casme2_dst_root)

    # SAMM 数据集
    # 路径配置
    samm_src_root = '/kaggle/input/samm-dataset/SAMM'
    samm_dst_root = '/kaggle/working/SAMM_onset_apex_offset'
    # 读取 Excel 标注文件
    samm_excel_path = '/kaggle/input/samm-dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'
    get_samm_onset_apex_offset(samm_src_root, samm_dst_root, samm_excel_path)
    zipPath = '/kaggle/working/SAMM_onset_apex_offset.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(samm_dst_root, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("SAMM关键帧目录结构如下：\n")
    print_directory_structure(samm_dst_root)
