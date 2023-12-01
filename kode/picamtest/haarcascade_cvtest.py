import cv2

cascPath = "/Users/kristianbotnen/OneDrive - University of Bergen/Presentasjoner/2023 - Pythonkurs avansert 2023/data/image_video/haarcascade_frontalface_default.xml"
#cascPath = "haarcascade_frontalcatface_extended.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def cascadetest(gray_frame, color_frame):
    
    detected_faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor=2, # Increase it to as much as 2 for faster detection, with the risk of missing some faces.
        minNeighbors=3, # Higher value results in less detections but with higher quality, 3~6 is a good.
        flags=cv2.CASCADE_SCALE_IMAGE
    )    
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(color_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return detected_faces

cap = cv2.VideoCapture(0)
# Change to a smaller framesize for speed
#ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
#ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,200)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascadetest(gray_frame, frame)
    #if len(faces) > 0:
        # Display the resulting frame
    cv2.imshow('frame', frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()