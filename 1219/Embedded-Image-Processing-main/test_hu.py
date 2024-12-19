import cv2
import numpy as np
from paddleocr import PaddleOCR

def adjust_contrast(image, alpha=1.3, beta=10):
    """
    調整圖像對比度與亮度
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def initialize_ocr():
    """
    初始化 PaddleOCR
    """
    return PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

def recognize_license_plate(image, ocr, x, y, w, h):
    """
    使用 PaddleOCR 辨識車牌號碼
    """
    plate_region = image[y:y+h, x:x+w]
    result = ocr.ocr(plate_region, cls=True)
    
    if result and result[0]:
        text = result[0][0][1][0]
        confidence = result[0][0][1][1]
        return text, confidence
    return None, 0

def enhance_and_detect_with_ocr(image_path):
    # 初始化 OCR
    ocr = initialize_ocr()
    
    # 讀取圖像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("無法讀取圖像，請檢查路徑！")
        return
    
    # 調整圖像大小
    image = cv2.resize(original_image, (1024, 800))
    result_image = image.copy()  # 創建副本以繪製結果
    
    # 步驟 1：增強對比度
    image_contrast = adjust_contrast(image, alpha=1.3, beta=10)
    cv2.imshow("Step 1: Adjusted Contrast", image_contrast)
    
    # 步驟 2：轉換為 HSV
    hsv = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2HSV)
    cv2.imshow("Step 2: HSV Image", hsv)
    
    # 步驟 3：檢測白色區域
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow("Step 3: White Mask", white_mask)
    
    # 步驟 4：形態學閉運算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Step 4: Closed Mask", closed_mask)
    
    # 步驟 5：尋找車牌候選區域並進行 OCR
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_plates = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        
        if 1000 < area < 200000 and 1.3 < aspect_ratio < 6.0 and h > 50:
            plate_text, confidence = recognize_license_plate(image, ocr, x, y, w, h)
            
            if plate_text:
                detected_plates.append({
                    'text': plate_text,
                    'confidence': confidence,
                    'position': (x, y, w, h)
                })
                
                # 繪製檢測結果
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f"{plate_text} ({confidence:.2f})",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 顯示結果
    print(f"\n檢測到 {len(detected_plates)} 個車牌：")
    for plate in detected_plates:
        print(f"車牌號碼: {plate['text']}, 置信度: {plate['confidence']:.2f}")
    
    # 顯示原圖與最終結果
    cv2.imshow("Original Image", image)
    cv2.imshow("License Plate Detection", result_image)
    cv2.imwrite('detection.jpg', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "pic/pic1.jpg"
    enhance_and_detect_with_ocr(image_path)
