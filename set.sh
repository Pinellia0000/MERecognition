# 不同环境需要修改
# source /root/miniconda3/bin/activate newCondaEnvironment
# sh /root/MERecognition/set.sh
pip install --upgrade pip
pip list --outdated
pip install --upgrade setuptools
pip list
python --version
pip --version

# 检测目前使用的python环境
which python
which pip

# 下载项目代码并安装依赖
rm MERecognition -rf
git clone https://github.com/Pinellia0000/MERecognition.git
# # 安装依赖
pip install -r /root/MERecognition/requirements.txt

# run
# python /root/MERecognition/get_onset_apex_offset.py
# python /root/MERecognition/crop_images.py
# python optflow_for_classify.py
#