# OTAS
Unsupervised Boundary Detection for Object-Centric Temporal Action Segmentation

### Prerequisites
conda create -n OTAS python=3.9
pip install opencv-python=4.7.0.72


### Data preparation
1. Download videos from [the breakfast dataset](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).
2. Extract frames and video info.
```
python video_info.py --dataset BF
```
3. Obtain object masks using pre-trained object detection model. [Detectron](https://github.com/facebookresearch/Detectron)
Save object class, box, and score for each video under /data/breakfast/object_info. 

### Train global perception model
```
python main.py --dataset BF --feature_model tf --seq_len 3 --num_seq 2 --mode train 
```
