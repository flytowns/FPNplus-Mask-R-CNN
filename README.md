# FPNplus-Mask-R-CNN

Wrote a paper accepted by ISICDM 2020 as the first author.

link: https://dl.acm.org/doi/fullHtml/10.1145/3451421.3451461

Abstract
----------------------------------------------------------------------------
In recent years, the number of lung cancer patients has continued to increase. In the process of detecting lung cancer, accurate segmentation of lung parenchyma plays a key role. In this paper, we proposed a method of lung parenchyma segmentation based on FPN++Mask R-CNN neural network model. The model improved original Mask R-CNN networks and optimized the structure of FPN (Feature Pyramid Networks), which is the feature extraction model of Mask R-CNN, by expanding the scale and level of FPN to fuse and extract more picture feature information from different levels. The experimental results show that compared with original Mask R-CNN models, FPN++Mask R-CNN demonstrates better segmentation results.

-----------------------------------------------------------------------------

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
