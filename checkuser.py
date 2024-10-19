import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
import ast  # Thư viện để chuyển đổi chuỗi thành mảng

# Đọc file CSV
df = pd.read_csv("User.csv")

# Chuyển đổi cột 'face_encoding' thành danh sách các mảng NumPy
Encode_List = df["face_encoding"].apply(lambda x: np.array(ast.literal_eval(x)))


def CHECK(Path):
    curImg = cv2.imread(Path)
    curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)

    # Mã hóa khuôn mặt của ảnh hiện tại
    EncodeCurImg = face_recognition.face_encodings(curImg)[0]

    # Tính khoảng cách giữa mã hóa hiện tại và các mã hóa đã lưu
    FaceDis = face_recognition.face_distance(Encode_List.tolist(), EncodeCurImg)

    # Tìm chỉ số của khuôn mặt gần nhất
    MatchIndex = np.argmin(FaceDis)

    # Kiểm tra xem khoảng cách có nhỏ hơn ngưỡng 0.5 không
    if FaceDis[MatchIndex] < 0.5:
        return df.iloc[MatchIndex]["id"]
    else:
        return "Không tìm thấy người dùng"


# print(CHECK("C:\\Users\\Asus\\Desktop\\BTL\\Pictest\\tl1.jpg"))
