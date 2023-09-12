import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pytesseract
import matplotlib.pyplot as plt

# Load mô hình đã huấn luyện
model_path = 'license_plate_recognition_model (1).h5'
model = load_model(model_path)


# Định nghĩa hàm để đọc biển số từ ảnh
def read_license_plate_number(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    plate_number = pytesseract.image_to_string(gray_plate, config='--psm 8')
    return plate_number.strip() if plate_number else None


# Định nghĩa hàm để kiểm tra biển số trên ảnh
def detect_license_plate(image, min_confidence=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    license_plates = license_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    best_plate_number = None
    best_plate_confidence = min_confidence

    for (x, y, w, h) in license_plates:

        print("x:", x)
        print("y:", y)
        print("w:", w)
        print("h:", h)

        license_plate_roi = image[y:y + h, x:x + w]
        resized_plate = cv2.resize(license_plate_roi, (64, 64))
        plate_gray = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
        plate_input = img_to_array(plate_gray)
        plate_input = np.expand_dims(plate_input, axis=0)
        plate_input = preprocess_input(plate_input)

        prediction = model.predict(plate_input)[0]

        if prediction > best_plate_confidence:
            best_plate_confidence = prediction
            best_plate_number = read_license_plate_number(license_plate_roi)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = f"{best_plate_number}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x
            text_y = y - 10 - text_size[1]
            #cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

    return image


# Đường dẫn đến ảnh từ camera
camera_image_path = 'caff.jpg'

# Đọc ảnh từ đường dẫn
camera_image = cv2.imread(camera_image_path)

# Kiểm tra và vẽ khung xanh quanh biển số (nếu có)
output_image = detect_license_plate(camera_image)

# Hiển thị ảnh đã xử lý
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
