# Instance Segmentation using Mask R-CNN on a Custom Dataset

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Authors
Arpit Aggarwal Shantam Bajpai


### Brief Introduction to the Project
For this project we will be addressing the task of Instance Segmentation, which combines object detection and semantic segmentation into a per-pixel object detection framework using a pre-trained Mask R-CNN model which will be fine tuned according to our dataset.


### Software Required
To run the jupyter notebooks and Python files, use Python 3. Standard libraries like numpy and PyTorch are used.


### Results obtained


### Dataset Description
The dataset used in this project is PedFudan Dataset. The number of classes are 2, the pedestrian and the background. The dataset can be found here.
 

### Steps for running the code
The training was done using the file "Code/train.py". The model was pre-trained on COCO dataset and consists of a Resnet-50 FPN which will act as a backbone to the Mask R-CNN Model and will output a feature map which is then fed into the Region Proposal Network that finds the image regions likely to contain objects. Mask R-CNN is built on top of Faster R-CNN, that is, it also provides segmentation masks for each instance. 


### Credits
The following links helped in completing the project:
1. https://github.com/jwyang/faster-rcnn.pytorch
2. https://github.com/multimodallearning/pytorch-mask-rcnn
