# 输出北京时间
TZ="Asia/Shanghai" date

# 查看系统CUDA版本
!nvidia-smi

# 查看linux系统信息
!cat /proc/version
!uname -a

# # ubuntu 22.04
!sudo apt-get update -y
!sudo apt-get upgrade -y

# 查看当前cuda版本
!nvcc -V

# 更新conda
!conda update -y -n base -c conda-forge conda

# Create New Conda Environment and Use Conda Channel
!conda create -y -n newCondaEnvironment python=3.10

# 不同环境需要修改
!source /root/miniconda3/bin/activate newCondaEnvironment

# 检测目前使用的python环境
!which python
!which pip

# 下载项目代码并安装依赖
!rm MERecognition -rf
!git clone https://github.com/Pinellia0000/MERecognition.git
# # 安装依赖
!pip install -r /kaggle/working/MERecognition/requirements.txt