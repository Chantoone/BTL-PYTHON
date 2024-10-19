import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
df = pd.read_csv("User.csv")
new_id = df['id'].max() + 1
def addnew(path):
    global df
    new_image=cv2.imread(path)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    new_encoding = face_recognition.face_encodings(new_image)[0]

    new_encoding_str = np.array2string(new_encoding, separator=', ')  # Chuyển đổi thành chuỗi
        # Tạo DataFrame mới cho người dùng mới
    new_row = pd.DataFrame({"face_encoding": new_encoding_str, "id": [new_id]})

        # Thêm hàng mới vào DataFrame hiện có
    df = pd.concat([df, new_row], ignore_index=True)

        # Lưu lại vào CSV
    df.to_csv("User.csv", index=False)
    print(f"Đã thêm người dùng mới với ID: {new_id}")

