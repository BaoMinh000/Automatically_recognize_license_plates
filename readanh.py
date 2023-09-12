# import các thư viện cần thiết
import cv2
import pytesseract
from PIL import Image

path_to_your_image='Screenshot 2023-08-27 194726.png'

# Chuyển hình ảnh đã chọn từ không gian màu BGR sang không gian màu xám
selected_image = cv2.imread(path_to_your_image)
gray_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)

# Áp dụng làm mờ Gaussian để giảm nhiễu
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Áp dụng phép nhị phân hóa Otsu để tách ký tự ra khỏi nền
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Chuyển hình ảnh nhị phân thành đối tượng PIL Image để sử dụng với pytesseract
binary_pil = Image.fromarray(binary)

# Sử dụng pytesseract để nhận dạng ký tự trong hình ảnh,
# sử dụng cấu hình tùy chỉnh để tối ưu cho việc nhận dạng ký tự
custom_config = r'--oem 3 --psm 6 outputbase digits'
recognized_characters = pytesseract.image_to_string(binary_pil, config=custom_config)

# Loại bỏ các khoảng trắng ở đầu và cuối chuỗi, nếu có
recognized_characters = recognized_characters.strip() if recognized_characters else None

print(recognized_characters)
