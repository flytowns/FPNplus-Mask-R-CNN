# FPNplus-Mask-R-CNN
对Mask R-CNN网络的FPN模块进行结构优化，提出了一个新的神经网络模型，命名为FPN++Mask R-CNN

核心改动模块：FPN(Feature Pyramid Networks)

FPN++Mask R-CNN 模型代码：Lung1/mrcnn/model3.py (FPN优化模块 见model3.py1910行)

原始Mask R-CNN 模型代码：Lung1/mrcnn/model.py 

文件夹说明：

mrcnn：存放神经网络模型

samples：训练测试模块；图像分割结果的显示模块；分割精度等指标的计算模块

除此之外，原文件中含有log文件和dataset文件，分别存放训练日志以及训练数据集

预权重使用 mask_rcnn_coco.h5 

------------------------------------------------------------------------------
Translation in English:

The structure of the FPN module of the Mask R-CNN network is optimized, and a new neural network model is proposed, named FPN++Mask R-CNN

Core module: FPN (Feature Pyramid Networks)

FPN++Mask R-CNN model code: Lung1/mrcnn/model3.py (FPN optimization module see model3.py1910 line)

Original Mask R-CNN model code: Lung1/mrcnn/model.py

Folder description:

mrcnn: store the neural network model

samples: training and testing module; display module for image segmentation results; calculation module for indicators such as segmentation accuracy

In addition, the original file contains log files and dataset files, which store training logs and training datasets respectively.

Pre-weights use mask_rcnn_coco.h5
