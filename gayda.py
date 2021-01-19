from peopletracker import PeopleTracker
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from beepy import beep
import numpy as np
import imutils
import time
import cv2
import pika
from detect_mask import mask_detection_image

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')
close = 0
objects_set = set()


def alert():
	beep(sound=1)


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# initialize our centroid tracker and frame dimensions
pt = PeopleTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = pt.update(rects)
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

	if len(objects) > 1:
		for i, (key_1, value_1) in enumerate(objects.items()):
			for j, (key_2, value_2) in enumerate(objects.items()):
				if i != j:
					object_1 = value_1
					object_2 = value_2
					if (object_1 is not None) and (object_2 is not None):
						xA = object_1[0]
						yA = object_1[1]
						xB = object_2[0]
						yB = object_2[1]
						D = dist.euclidean((xA, yA), (xB, yB))
						(mX, mY) = midpoint((xA, yA), (xB, yB))
						if D < 150.0:
							cv2.line(frame, (object_1[0], object_1[1]), (object_2[0], object_2[1]), (139,0,0),
									thickness=1, lineType=8)
							cv2.putText(frame, "{:.1f}".format(D), (int(mX), int(mY - 10)),
										cv2.FONT_HERSHEY_SIMPLEX, 0.55, (139,0,0), 2)

							body_str = str(key_1) + "," + str(key_2)
							if(key_1 not in objects_set or key_2 not in objects_set):
								channel.basic_publish(exchange='',
													routing_key='hello',
													body = body_str)
								objects_set.add(key_1)
								objects_set.add(key_2)
						else:
							cv2.line(frame, (object_1[0], object_1[1]), (object_2[0], object_2[1]), (0, 255, 0),
									thickness=1, lineType=8)
							cv2.putText(frame, "{:.1f}".format(D), (int(mX), int(mY - 10)),
										cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 0, 159), 2)
							objects_set.clear()
					else:
						break

	people_count = len(objects)
	height, width, channels = frame.shape
    frame = mask_detection_image(frame)

	# show the output frame
    cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
connection.close()