# Audio-Visual-Question-Answering-AVQA
This task is based on MUSIC-AVQA Dataset.  
We focus on optimizing the accuracy of AVQA task, which aims to answer questions regarding different visual objects, sounds, and their associations in videos. The problem requires comprehensive multimodal understanding and spatio-temporal reasoning over audio-visual scenes.
## Dataset
We use the dataset of MUSIC-AVQA Dataset, which is released in the CVPR 2022 paper:  
Learning to Answer Questions in Dynamic Audio-Visual Scenarios (Oral Presentation).  
The dataset can be downloaded at [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/).
## Preparation of the pretrained models
The resnet-18 and vggish networks can be downloaded at [Res-18](https://pan.baidu.com/s/1v4DAdjp0o2atNiKknh2H4w?pwd=2g8i 提取码: 2g8i), [VGGish model](https://pan.baidu.com/s/1VUyfTGPvlv_sJzLPw5Kc2g?pwd=r4of 提取码: r4of), and [VGGish params](https://pan.baidu.com/s/1eoE8AhEsch_-_RUfhfZwAQ?pwd=r3hf 提取码: r3hf). These models should be placed at /data/pretrained_model folder.
## Code
```
/models includes the backbone and other supporting modules we use
/dataloader includes the dataloader for data prepocessing
/data includes all required data and pretrained models to extract features
```
Our implement is based on [MUSIC-AVQA](https://pages.github.com/) and [MIMN](https://github.com/xunan0812/MIMN). Thanks them for their contribution!
## Preliminary optimization results
![image](https://github.com/zailongchen/Audio-Visual-Question-Answering-AVQA/blob/main/result%20images/result_0.png)
