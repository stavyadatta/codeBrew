import numpy as np
import argparse
import cv2
from imutils import paths
import os
import imutils

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--input', required=True, help='putting in image')
# ap.add_argument('-p', '--prototxt', default='deploy.prototxt.txt')
# ap.add_argument('-m', '--caffemodel', default='res10_300x300_ssd_iter_140000.caffemodel')
# ap.add_argument('-c', '--confidence', default=0.5)
# ap.add_argument('-o', '--output', default='output/test.jpg')

# args = vars(ap.parse_args())

detector = cv2.dnn.readNetFromCaffe("face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

image_paths = list(paths.list_images('dataset/with_mask'))
imageNum = 0

for (i, imagePath) in enumerate(image_paths):
    
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), 
                                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            try:
                cv2.imwrite(f"new_dataset/with_mask/test{imageNum}.jpg", face)
            except:
                continue
            imageNum += 1
            print(imageNum)