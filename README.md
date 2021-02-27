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
