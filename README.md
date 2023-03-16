# OTAS
Unsupervised Boundary Detection for Object-Centric Temporal Action Segmentation

### Prerequisites
conda create -n OTAS python=3.9
pip install opencv-python=4.7.0.72
pip install torch_geometric
pip install torch_sparse


### Data preparation
1. Download videos from [the breakfast dataset](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).
3. Extract frames and video info.
```
cd code
python video_info.py --dataset BF
```
3. Obtain object info using pre-trained object detection model. We provide example object info file from the platform [Detectron](https://github.com/facebookresearch/Detectron) using Faster-RCNN-X101-FPN model pre-trained on the COCO train2017 dataset. The object info file should contain object class, box and score. 

### Train global perception model
```
python main.py --dataset BF --feature_model tf --seq_len 3 --num_seq 2 --mode train 
```
