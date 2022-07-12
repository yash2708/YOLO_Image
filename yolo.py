import numpy as np
import time
import cv2

def detect_object(image):
    LABELS_FILE='Yolo/data/coco.names'
    CONFIG_FILE='Yolo/cfg/yolov3-tiny.cfg'
    WEIGHTS_FILE='Yolo/yolov3-tiny.weights'
    CONFIDENCE_THRESHOLD=0.3
    LABELS = open(LABELS_FILE).read().strip().split("\n")
    np.random.seed(4)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
	    for detection in output:
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]
		    if confidence > CONFIDENCE_THRESHOLD:
			    box = detection[0:4] * np.array([W, H, W, H])
			    (centerX, centerY, width, height) = box.astype("int")
			    x = int(centerX - (width / 2))
			    y = int(centerY - (height / 2))
			    boxes.append([x, y, int(width), int(height)])
			    confidences.append(float(confidence))
			    classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,CONFIDENCE_THRESHOLD)
    if len(idxs) > 0:
	    for i in idxs.flatten():
		    (x, y) = (boxes[i][0], boxes[i][1])
		    (w, h) = (boxes[i][2], boxes[i][3])
		    color = [int(c) for c in COLORS[classIDs[i]]]
		    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image
