# sh /root/MERecognition/start.sh
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

# 查看当前cuda版本
nvcc -V

# 更新conda
conda update -y -n base -c conda-forge conda
# Create New Conda Environment and Use Conda Channel
conda create -y -n newCondaEnvironment python=3.10
