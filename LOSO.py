import os
import shutil

# data_folder 只包含关键帧和光流图像
# 在LOSO划分后 直接用于训练
# data_folder = 'Dataset/CASME2_retinaface'
data_folder = '/kaggle/working/CASME2_retinaface'

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
