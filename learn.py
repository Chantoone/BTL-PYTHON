import cv2

def compare_book_covers_orb(image_path1, image_path2):
    # Đọc ảnh
    img1 = cv2.imread(image_path1, 0)  # Ảnh bìa sách cần kiểm tra
    img2 = cv2.imread(image_path2, 0)  # Ảnh bìa sách trong cơ sở dữ liệu

    # Khởi tạo ORB
    orb = cv2.ORB_create()

    # Phát hiện và tính toán keypoints, descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Sử dụng BFMatcher với NORM_HAMMING
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # So khớp các descriptors
    matches = bf.match(des1, des2)

    # Sắp xếp các matches theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)

    # Tính toán tỷ lệ khớp dựa trên số matches
    match_percentage = len(matches) / max(len(kp1), len(kp2))

    print(f'Tỷ lệ khớp: {match_percentage * 100}%')

    return match_percentage > 0.7  # Giả định tỷ lệ khớp > 70% là khớp
print(compare_book_covers_orb("C:\\Users\\Asus\\Desktop\BTL\\hdh2.jpg", "C:\\Users\\Asus\\Desktop\\BTL\\hdh1.jpg"))