# Audio-Visual-Question-Answering-AVQA
This task is based on MUSIC-AVQA Dataset.  
We focus on optimizing the accuracy of AVQA task, which aims to answer questions regarding different visual objects, sounds, and their associations in videos. The problem requires comprehensive multimodal understanding and spatio-temporal reasoning over audio-visual scenes.
## Dataset
We use the dataset of MUSIC-AVQA Dataset, which is released in the CVPR 2022 paper:  
Learning to Answer Questions in Dynamic Audio-Visual Scenarios (Oral Presentation).  
The dataset can be downloaded at [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/).
## Code
```
/models includes the backbone and other supporting modules we use
/dataloader includes the dataloader for data prepocessing
/data includes all required data and pretrained models to extract features
```
Our implement is based on [MUSIC-AVQA](https://pages.github.com/) and [MIMN](https://github.com/xunan0812/MIMN). Thanks them for their contribution!
## Preliminary optimization results
![image](https://github.com/zailongchen/Audio-Visual-Question-Answering-AVQA/blob/main/image/results_1.png)
