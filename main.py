import cv2
import face_recognition
imgtl=face_recognition.load_image_file("pic/tl1.jpg")
imgtl=cv2.cvtColor(imgtl,cv2.COLOR_BGR2RGB)

imgcheck=face_recognition.load_image_file("pic/tl2.jpg")
imgcheck=cv2.cvtColor(imgcheck,cv2.COLOR_BGR2RGB)

"""xác định vị trí khuôn mặt"""
faceLoc=face_recognition.face_locations(imgtl)[0]#y1,x2,y2,x1
print(faceLoc)
#ma hoa hinh anh
encodeTL=face_recognition.face_encodings(imgtl)[0]
cv2.rectangle(imgtl,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

facecheck=face_recognition.face_locations(imgcheck)[0]
encodeCheck=face_recognition.face_encodings(imgcheck)[0]
cv2.rectangle(imgcheck,(facecheck[3],facecheck[0]),(facecheck[1],facecheck[2]),(0,255,0),2)

res=face_recognition.compare_faces([encodeTL],encodeCheck)

faceDis=face_recognition.face_distance([encodeTL],encodeCheck)
cv2.putText(imgcheck,f"{res}{1-round(faceDis[0],2)}",(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
print(faceDis)
print(res)
cv2.imshow("TL1",imgtl)
cv2.imshow("TL2",imgcheck)
cv2.waitKey()
