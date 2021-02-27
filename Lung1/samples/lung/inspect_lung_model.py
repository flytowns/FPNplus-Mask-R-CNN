#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Lung Trained Model
# 
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.lung import lung

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights
LUNG_WEIGHTS_PATH = os.path.join(ROOT_DIR,'mask_rcnn_lung.h5')


# ## Configurations

# In[2]:


config = lung.LungConfig()
LUNG_DIR = os.path.join(ROOT_DIR, "datasets/lung")


# In[3]:


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Notebook Preferences

# In[4]:


# use CPU and leave the GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
TEST_MODE = "inference"


# In[5]:


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Load Validation Dataset

# In[6]:


# Load validation dataset
dataset = lung.LungDataset()
dataset.load_lung(LUNG_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# ## Load Model

# In[7]:


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# In[8]:


# Set path to lung weights file
weights_path = LUNG_WEIGHTS_PATH 
# weights_path = model.find_last() # use the last trained weights file

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# ## Run Detection

# In[10]:


for image_id in range(5):
    #image_id = random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]

# since input image has 16 bits, convert to 8 bits first just for visualize
image_visualize = np.int8(image/(2**8))
visualize.display_instances(image_visualize, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)


# ## compute Dice

# In[12]:


# get metrics for one sample
def getDice(seg, gt):    # seg and gt are both boolen array
    
    smooth = 1 # avoid division by zero

    # compute intersection and union for calculating metrics
    
    intersection = np.sum(np.multiply(seg,gt)) # element-wise product
    aa = 2*intersection + smooth
    bb = np.sum(seg) + np.sum(gt) + smooth
    
    Dice = aa / bb
    Rs = np.sum(gt)
    Os = np.sum(seg) - intersection
    Us = Rs - intersection
    OR = (Os + smooth) / (Rs + Os + smooth)
    UR = (Us + smooth) / (Rs + Os + smooth)
    
    return (Dice,OR,UR)


# In[13]:


# get ground truth and segmentation result for one image
def pred(image_path):
    
    # load image
    image = dataset.load_image(image_path)
    mask, class_ids = dataset.load_mask(image_path)
    original_shape = image.shape
    
    # resize
    image, window, scale, padding, _ = utils.resize_image(
    image, 
    min_dim=config.IMAGE_MIN_DIM, 
    max_dim=config.IMAGE_MAX_DIM,
    mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    
    # get segmentation results
    results = model.detect([image], verbose=1)
    mask_pred = results[0]["masks"]


# In[ ]:


#im_path =

