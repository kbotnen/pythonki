import cv2 # pip install opencv-contrib-python
import numpy as np

#Labels of network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# https://github.com/djmv/MobilNet_SSD_opencv
proto_file = '/Users/kristianbotnen/Library/CloudStorage/OneDrive-UniversityofBergen/Presentasjoner/2023 - Pythonkurs avansert 2023/data/image_video/MobileNetSSD_deploy.prototxt'
model_file = '/Users/kristianbotnen/Library/CloudStorage/OneDrive-UniversityofBergen/Presentasjoner/2023 - Pythonkurs avansert 2023/data/image_video/MobileNetSSD_deploy.caffemodel'

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame_resized = cv2.resize(frame, (300,300)) # Resize the frame for prediction
    input_shape = (300, 300)
    mean = (127.5, 127.5, 127.5)
    scale = 0.007843
    # After executing this command our "blob" now has the shape: (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(frame_resized, scalefactor=scale, size=input_shape, mean=mean, swapRB=True) 
    
    # Set to network the input blob 
    net.setInput(blob)
    
    # Prediction of network
    detections = net.forward()
    
    # Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    # For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] # Confidence of prediction 
        if confidence > 0.7: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                #print(detections[0,0,i,:])

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Done")
