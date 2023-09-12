import cv2

image = 'caff.jpg'
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    license_plates = license_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
def read_license_plate_number(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    plate_number = pytesseract.image_to_string(gray_plate, config='--psm 8')
    return plate_number.strip() if plate_number else None

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
    # cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
