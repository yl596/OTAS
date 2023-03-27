# OTAS
Unsupervised Boundary Detection for Object-Centric Temporal Action Segmentation
![eed264b0-2b7e-47df-ab72-ab4eef3e8de5_1](https://user-images.githubusercontent.com/67664107/227826095-e31225f6-8765-46a3-9f6d-95da3d6dc355.jpeg)
![model](https://user-images.githubusercontent.com/67664107/227825145-e7a463e8-d3e7-4565-a3cc-c60bdd2d7a76.jpg)

### Prerequisites
conda create -n OTAS python=3.9
pip install opencv-python==4.7.0.72
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric
pip install torch_sparse
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install pandas

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
