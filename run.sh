# # sh /root/MERecognition/run.sh

rm MERecognition -rf
git clone https://github.com/Pinellia0000/MERecognition.git

python /root/MERecognition/get_onset_apex_offset.py

python /root/MERecognition/crop_images.py

python /root/MERecognition/optflow_for_classify.py

python /root/MERecognition/emotion_for_loso.py

python /root/MERecognition/LOSO.py