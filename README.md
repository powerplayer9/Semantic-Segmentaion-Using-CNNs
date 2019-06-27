# Semantic-Segmentaion-Using-CNNs
Analyzing the performance of different types of convolutional filters for image segmentation purposes.

## Experimental Setup
Dataset: CityScapes

Input Resolution: 512 x 256 x 3

GPU: NVIDIA Tesla K20c

Gradient Update Method: 

First 10 Epoch:AdaGrad

After 10 Epoch:SGD

## Results
mIoU on Test Dataset

| Model  | mIoU | 
| ------------- | ------------- |
| FCN8  | 51.1  |
| DUC-HDC  | 28.5  |
| U-Net  | 23.4  |
