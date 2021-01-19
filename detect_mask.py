from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    try:
        (h, w) = frame.shape[:2]
    except AttributeError:
        return (None, None)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs, preds)

def mask_detection_video(input_video_path, output_video_path, face_detector="face_detector", model="mask_detector_kaggle.model"):
    print("[INFO] loading face detector model")
    prototxtPath = os.path.sep.join([face_detector, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_detector,
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    print("[INFO] loading face mask detector model")
    maskNet = load_model(model)
    
    print("[INFO] getting video from input")
    vid = cv2.VideoCapture(input_video_path)
    writer = None
    
    # frame count
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    totalFrames = int(vid.get(prop))
    print(f"info - Total frames are {totalFrames}")
    
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc('M','J','P','G')
    # print(output_video_path)
    # out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
    i = 0
    while True:
        _, frame = vid.read()
        print("hahahahaaha")       
        
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        frame = original_frame
        frame = imutils.resize(frame, width=400)
        
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        if locs == None and preds == None:
            return
        
        
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('output_videos/test.avi', fourcc, 30, 
                                    (frame.shape[1], frame.shape[2]), True)
        # writer.write(frame)
        
        cv2.imwrite("output_videos/test" + str(i) + ".jpg", frame)
        
        i+=1
        
    # writer.release()
    # vid.release()
    cv2.destroyAllWindows()

def mask_detection_image(frame, face_detector="face_detector", model="mask_detector_keras_new_dataset.model"):

    print("[INFO] loading face detector model")
    prototxtPath = os.path.sep.join([face_detector, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_detector,
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    #frame = cv2.imread(frame)
    
    print("[INFO] loading face mask detector model")
    maskNet = load_model(model)
    frame = imutils.resize(frame, width = 400)
    
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    if locs == None and preds == None:
        return
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        cv2.imwrite("output_images/test.jpg", frame)
    return frame 
        
      
    


# input_video_path = "input_videos/both.mp4"
# output_video_path = 'output_videos/test.avi'
# input_image_path = "input_images/noMask.jpg"

# mask_detection_video(input_video_path, output_video_path)            
# mask_detection_image(input_image_path)
