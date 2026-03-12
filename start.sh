# rm MERecognition -rf
# git clone https://github.com/Pinellia0000/MERecognition.git
# sh /root/autodl-tmp/MERecognition/start.sh
# 输出北京时间
TZ="Asia/Shanghai" date

# 查看系统CUDA版本
nvidia-smi

# 查看linux系统信息
cat /proc/version
uname -a

# # ubuntu 22.04
sudo apt-get update -y
sudo apt-get upgrade -y

# rm /root/autodl-tmp/working -rf
# 查看当前cuda版本
nvcc -V

# 更新conda
conda clean -a -y
conda update -y -n base -c conda-forge conda
# 原环境报错
python -m pip install --upgrade pip
python -m pip install setuptools

# 环境正在使用 退出环境
# conda deactivate
# 删除环境
conda remove -n newCondaEnvironment --all
# 确认是否删除完成
conda env list
# Create New Conda Environment and Use Conda Channel
conda create -y -n newCondaEnvironment python=3.10
