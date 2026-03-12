# sh /root/autodl-tmp/MERecognition/run.sh

#rm MERecognition -rf
#git clone https://github.com/Pinellia0000/MERecognition.git
# rm /root/autodl-tmp/working -rf

rm /root/autodl-tmp/working -rf
rm /root/autodl-tmp/MERecognition/logs/test1 -rfcd

# 运行前每次必修改 并进行替换
mkdir -p /root/autodl-tmp/MERecognition/logs/test1

python -u /root/autodl-tmp/MERecognition/get_onset_apex_offset.py 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_get_onset_apex_offset.txt

python -u /root/autodl-tmp/MERecognition/crop_images.py 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_crop_images.txt

python -u /root/autodl-tmp/MERecognition/optflow_for_classify.py 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_optflow_for_classify.txt

python -u /root/autodl-tmp/MERecognition/emotion_for_loso.py 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_emotion_for_loso.txt

python -u /root/autodl-tmp/MERecognition/LOSO.py 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_LOSO.txt

# # CASME2 3分类
python -u /root/autodl-tmp/MERecognition/train_classify_SKD_TSTSAN.py --train True --main_path "/root/autodl-tmp/working/CASME2_retinaface_loso_3" --exp_name "exp_CASME2_3" --class_num 3 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_CASME2_retinaface_loso_3.txt

# # CASME2 5分类
python -u /root/autodl-tmp/MERecognition/train_classify_SKD_TSTSAN.py --train True --main_path "/root/autodl-tmp/working/CASME2_retinaface_loso_5" --exp_name "exp_CASME2_5" --class_num 5 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_CASME2_retinaface_loso_5.txt

# # SAMM 3分类
python -u /root/autodl-tmp/MERecognition/train_classify_SKD_TSTSAN.py --train True --main_path "/root/autodl-tmp/working/SAMM_retinaface_loso_3" --exp_name "exp_SAMM_3" --class_num 3 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_SAMM_retinaface_loso_3.txt

# # CASME3 3分类
python -u /root/autodl-tmp/MERecognition/train_classify_SKD_TSTSAN.py --train True --main_path "/root/autodl-tmp/working/CASME3_retinaface_loso_3" --exp_name "exp_CASME3_3" --class_num 3 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_CASME3_retinaface_loso_3.txt

# # CASME3 4分类
python -u /root/autodl-tmp/MERecognition/train_classify_SKD_TSTSAN.py --train True --main_path "/root/autodl-tmp/working/CASME3_retinaface_loso_4" --exp_name "exp_CASME3_4" --class_num 4 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_CASME3_retinaface_loso_4.txt

# # CASME3 7分类
python -u /root/autodl-tmp/MERecognition/train_classify_SKD_TSTSAN.py --train True --main_path "/root/autodl-tmp/working/CASME3_retinaface_loso_7" --exp_name "exp_CASME3_7" --class_num 7 2>&1 | tee /root/autodl-tmp/MERecognition/logs/test1/log_CASME3_retinaface_loso_7.txt

## 本地终端执行
## 注意改名称 与前面名称保持一致
#scp -P 29211 -r root@connect.nmb1.seetacloud.com:/root/autodl-tmp/MERecognition/logs/test1 "D:\PycharmProjects\MERecognition\logs"