"""Detection augmentation from jianchong."""

#coding=utf-8
# change from PIL to cv2
import random

import torch
import cv2
from torchvision.transforms import functional as F
import numpy as np 


class DetColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness_range: How much to jitter brightness.  Should be non negative numbers.
        contrast_range: How much to jitter contrast. Should be non negative numbers.
        saturation_range: How much to jitter saturation. Should be non negative numbers.
        hue_range: How much to jitter hue. Should have -0.5 <= min <= max <= 0.5.
    """    
    def __init__(self, 
                prob=0.5,
                brightness_delta=32, 
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18
    ):
        self.prob = prob

        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, image, target=None):
        """
        Args:
            image (cv2 Image): Input image.
        Returns:
            cv2 Image: Color jittered image.
        """
        image = np.array(image)
        if random.random() > self.prob:
            return image
        else:
            image = image.astype(np.float32)

            # random brightness
            if random.randint(0, 2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                image += delta
            
            # mode == 0 --> do random contrast last
            # mode > 0  --> do random contrast first
            mode = random.randint(0, 2)
            if mode > 0:
                # random contrast
                if random.randint(0, 2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    image *= alpha
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # random saturation
            if random.randint(0, 2):
                image[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)
            
            # random hue
            if random.randint(0, 2):
                image[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                image[..., 0][image[..., 0] > 360] -= 360
                image[..., 0][image[..., 0] < 0] += 360

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            if mode == 0:
                # random contrast
                if random.randint(0, 2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    image *= alpha
            
            # randomly swap channels
            if random.randint(0, 1):
                image = image[..., np.random.permutation(3)]

            return image.astype('uint8')
