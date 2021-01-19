#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:02:11 2020

@author: ghaidaa
"""
import cv2
from detect_mask import mask_detection_image



def frame_show(frame):
    frame = mask_detection_image(frame)
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    return key
    
