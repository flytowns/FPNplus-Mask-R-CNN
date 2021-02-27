#!/usr/bin/env python
# coding: utf-8

# In[2]:


# set GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import tensorflow as tf
config = tf.ConfigProto()  
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)


from flyai.utils.log_helper import train_log


from flyai.train_helper import upload_data, download, sava_train_model

download("Lung.zip",decompression=True)

# In[3]:


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw


# In[4]:


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


# In[5]:


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# In[6]:


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# In[7]:


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
command = 'train'
dataset = '../../../Lung/datasets/lung'
weights = '../../../Lung/mask_rcnn_lung.h5'
logs = DEFAULT_LOGS_DIR
image = ''
video = ''

# ##  Configurations

# In[8]:


class LungConfig(Config):
    """Configuration for training on the lung dataset.
    Derives from the base Config class and overrides some values.
    """
    
    NAME = "lung"

    # batchsize = IMAGES_PER_GPU*GPU_COUNT
    IMAGES_PER_GPU = 1
    #GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + lung

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



# ## Dataset

# In[9]:


class LungDataset(utils.Dataset):

    def load_lung(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("lung", 1, "lung")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Walk all samples under the current dataset
        all_image_samples = os.listdir(os.path.join(dataset_dir,'images'))
        for image_sample in all_image_samples:
            image_path = os.path.join(dataset_dir,'images',image_sample)
            
            # Parse JSON to get annotation
            json_path = image_path.replace('images','annotations').replace('png','json').replace('tif','json')
            annotation = json.load(open(json_path,'rb'))
            polygons = [a for a in annotation["shapes"]] # list of polygons for one image
            
            # Add the height and weight for the current image
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            self.add_image(
                "lung",
                image_id=image_sample,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
            
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "lung":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            points = np.array(p["points"])
            all_points_y = list(points[:,1])
            all_points_x = list(points[:,0])
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
            
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lung":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# ## Train

# In[10]:


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LungDataset()
    dataset_train.load_lung(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LungDataset()
    dataset_val.load_lung(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=1e-4,
                epochs=100,
                layers='heads')


# ## Splash effects

# In[16]:


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


# In[17]:


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
      #  print("Running on {}".format(image))
        # Read image
        image = skimage.io.imread()
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


# ## Training

# In[18]:


if __name__ == '__main__':
    import argparse

    # Parse command line arguments

    '''
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect lungs.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/lung/dataset/",
                        help='Directory of the Lung dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
    '''
    # Validate arguments
    if command == "train":
        assert dataset, "Argument --dataset is required for training"
    elif command == "splash":
        assert image or video,               "Provide --image or --video to apply color splash"

    print("Weights: ", weights)
    print("Dataset: ", dataset)
    print("Logs: ", logs)

    # Configurations
    if command == "train":
        config = LungConfig()
    else:
        class InferenceConfig(LungConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if command == "train":
        train(model)
    elif command == "splash":
        detect_and_color_splash(model, image_path=image,
                                video_path=video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(command))

    sava_train_model(model_file="../../mask_rcnn_lung.h5", overwrite=False)

# 使用文件加路径名即可下载
    download("../../mask_rcnn_lung.h5")
# In[ ]:




