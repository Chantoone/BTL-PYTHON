import cv2
import os


def resize_images_in_directory(directory_path, image_size=(224, 224)):
    # Duyệt qua từng class folder trong thư mục
    for class_folder in os.listdir(directory_path):
        class_path = os.path.join(directory_path, class_folder)

        # Kiểm tra nếu class_path là một thư mục
        if os.path.isdir(class_path):
            # Duyệt qua từng ảnh trong thư mục class
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                # Đọc ảnh
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Kiểm tra nếu ảnh đọc thành công
                if image is not None:
                    # Thay đổi kích thước ảnh thành 224x224
                    resized_image = cv2.resize(image, image_size)

                    # Ghi đè ảnh mới lên ảnh cũ
                    cv2.imwrite(image_path, resized_image)
                else:
                    print(f"Lỗi khi đọc ảnh: {image_path}")


# Thay đổi kích thước ảnh trong thư mục 'train'
# resize_images_in_directory("data/train")

# Thay đổi kích thước ảnh trong thư mục 'validation'
resize_images_in_directory("data/validation")
