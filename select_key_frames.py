import os
import shutil
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库


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


def select_key_frames():
    """
    从数据集中选取起始帧、顶点帧和结束帧这样的关键帧
    关于标签文件中有的帧没有标注的处理办法：跳过该样本
    """
    # 路径配置
    src_root = '/kaggle/input/casmeii/CASME2-RAW/CASME2-RAW'
    dst_root = '/kaggle/working/CASME2_key_frames'
    os.makedirs(dst_root, exist_ok=True)

    # 读取 Excel 标注文件
    # sub04 EP12_01f 的顶点帧在注释文件中没有给出 标记为/
    excel_path = '/kaggle/input/casmeii/CASME2-coding-20140508.xlsx'
    df = pd.read_excel(excel_path)

    # 遍历每一行数据（每个视频一个条目），使用 tqdm 包装迭代器以显示进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        subject = str(row['Subject']).strip()  # e.g., '01'
        filename = str(row['Filename']).strip()  # e.g., 'EP02_01f'
        onset = safe_parse_int(row['OnsetFrame'])
        apex = safe_parse_int(row['ApexFrame'])
        offset = safe_parse_int(row['OffsetFrame'])

        # 跳过任意帧信息无效的样本
        if None in (onset, apex, offset):
            print(f"[SKIP] Invalid frame data for: {filename}")
            continue

        # 构造视频所在的路径，如 sub01/EP02_01f/
        # 注意：原 Excel 中 Subject 字段已经是两位数，一般不需要再 zfill，但此处保留兼容性
        video_folder = os.path.join(src_root, f"sub{subject.zfill(2)}", filename)

        for frame_type, frame_id in [('onset', onset), ('apex', apex), ('offset', offset)]:
            # 注意 CASME II 中的图片名格式：img1.jpg之类 不需要填充0来补充位数
            img_name = f"img{frame_id}.jpg"
            src_img_path = os.path.join(video_folder, img_name)

            if os.path.exists(src_img_path):
                # 为防冲突 （如多个视频都有img1.jpg）
                # 会保存为类似 EP02_01f_onset.jpg EP02_01f_apex.jpg EP02_01f_offset.jpg
                dst_img_name = f"{filename}_{frame_type}.jpg"
                dst_img_path = os.path.join(dst_root, dst_img_name)
                shutil.copy(src_img_path, dst_img_path)
                print(f"Copied: {src_img_path} → {dst_img_path}")
            else:
                print(f"[WARNING] Not found: {src_img_path}")


if __name__ == "__main__":
    select_key_frames()
