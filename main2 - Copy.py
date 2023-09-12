mport cv2
import numpy as np
from easyocr import easyocr
from pytesseract import pytesseract
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import re


from PIL import Image
import easyocr
# Load mô hình đã huấn luyện
model_path = 'license_plate_recognition_model (1).h5'
model = load_model(model_path)

# Định nghĩa hàm để đọc biển số từ ảnh

def read_license_plate_number(license_plates):
    # Chuyển hình ảnh biển số từ không gian màu BGR sang không gian màu xám
    gray_plate = cv2.cvtColor(license_plates, cv2.COLOR_BGR2GRAY)

    # Tạo một chuỗi chứa tất cả các chữ cái in hoa, số và một số ký tự đặc biệt
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"

    # Tăng độ phân giải hình ảnh
    license_plates = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Tăng độ tương phản giữa ký tự và nền
    license_plates = cv2.equalizeHist(license_plates)

    # Sử dụng thuật toán nhận dạng ký tự để đọc các ký tự trong hình ảnh
    plate_number = pytesseract.image_to_string(license_plates, config='--psm 6 -c tessedit_char_whitelist=' + whitelist)
    print(plate_number)

    # Kiểm tra xem plate_number có phù hợp với cấu trúc mong muốn hay không
    if re.match(r'^([0-9]{2}[A-Z]{1} - [0-9]{3}\.[0-9]{2}'
                r'|[0-9]{2}[A-Z]{1}-[0-9]{5}' # không dấu cách 
                r'|[0-9]{1,2}[A-Z]{1} - [0-9]{5}'# có dấu cách ở '-'
                r'|[0-9]{2}[A-Z]{1}-[0-9]{3}\.[0-9]{2})$'# không dấu cách
            , plate_number):
        # Loại bỏ các khoảng trắng ở đầu và cuối chuỗi, nếu có
        return plate_number.strip()
    else:
        return None

# Định nghĩa hàm để kiểm tra biển số trên ảnh
def detect_license_plate(image, min_confidence=0.8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    license_plates = license_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    best_plate_number = None
    best_plate_confidence = min_confidence

    h, w = 0, 0
    license_plate_roi = None

    for (x, y, w, h) in license_plates:
        license_plate_roi = image[y:y + h, x:x + w]

        # Kiểm tra chất lượng hình ảnh
        gray_roi = cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if fm < 50:  # Đây là giá trị ngưỡng mà bạn có thể điều chỉnh
            continue  # Nếu hình ảnh mờ, bỏ qua và không xử lý nó

        resized_plate = cv2.resize(license_plate_roi, (64, 64))
        plate_gray = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
        plate_input = img_to_array(plate_gray)
        plate_input = np.expand_dims(plate_input, axis=0)
        plate_input = preprocess_input(plate_input)

        prediction = model.predict(plate_input)[0]

        if prediction > best_plate_confidence:
            best_plate_confidence = prediction
            best_plate_number = read_license_plate_number(license_plate_roi)

            # Khởi tạo biến để lưu biển số xe được phát hiện cuối cùng
            last_detected_plate_number = None
            # Giả sử rằng `new_plate_number` là biển số xe mới được phát hiện
            new_plate_number = read_license_plate_number(license_plate_roi)

            #In ra khung khi phat hien vung chua bien so
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Nếu có một biển số xe mới và nó khác với biển số xe cuối cùng được phát hiện
            if new_plate_number and new_plate_number != last_detected_plate_number:
                last_detected_plate_number = new_plate_number

                # In số biển số ra cửa sổ dòng lệnh
                print(f"Detected license plate number: {last_detected_plate_number}")



                text = f"{last_detected_plate_number}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = x
                text_y = y - 10 - text_size[1]
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

                # Hiển thị hình ảnh
                cv2.imshow('License Plate', image)

                # Chờ 1ms trước khi cập nhật khung hình tiếp theo
                cv2.waitKey(1)

    return image, h, w, license_plate_roi




# Mở camera máy tính
cap = cv2.VideoCapture(0)

frame_count = 0
last_plate_image = None

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()

    # Kiểm tra và vẽ khung xanh quanh biển số (nếu có)
    output_frame, h, w, license_plate_roi = detect_license_plate(frame)

    # Hiển thị khung hình đã xử lý lên màn hình
    cv2.imshow('License Plate Detection', output_frame)

    # Cập nhật hình ảnh vùng chứa biển số sau mỗi 5 khung hình
    frame_count += 1
    if frame_count % 5 == 0:
        last_plate_image = license_plate_roi

    # Hiển thị hình ảnh vùng chứa biển số cuối cùng đã chụp
    if last_plate_image is not None:
        cv2.imshow('Last Captured License Plate', last_plate_image)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# Khi hoàn thành, giải phóng camera và đóng tất cả cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()


