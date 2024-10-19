import cv2
import face_recognition
import os
import numpy as np
path="pic"
images=[]
classNames=[]
myList=os.listdir(path)
for cl in myList:
    curImg=cv2.imread(os.path.join(path,cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(len(images))
print(classNames)

def mahoa(images):
    encodeList=[]
    for image in images:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(image)[0])
    return encodeList
encodeListKnow=mahoa(images)
print("Ma hoa thanh cong")
print(len(encodeListKnow))

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    framS=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    framS=cv2.cvtColor(framS,cv2.COLOR_BGR2RGB)

    #xac dinh vi tri hien tại trên cam
    facecurFrame=face_recognition.face_locations(framS)
    encodecurFrame=face_recognition.face_encodings(framS)
    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame):
        matches=face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnow,encodeFace)
        matchIndex=np.argmin(faceDis)
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
        else:
            name = "Unknow"
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("camera",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()