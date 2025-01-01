import cv2
import numpy as np
from paddleocr import PaddleOCR

def cutVideo():
    i = 0
    video = cv2.VideoCapture('video/video.avi')
    while(True):
        ret,frame = video.read()
        cv2.imshow('video',frame)
        c = cv2.waitKey(50)
        if c == 27:
            break
        i=i+1
        if i%6==0:
            cv2.imwrite('video/pic/pic'+str(i)+'.jpg',frame)

def adjust_contrast(image, alpha, beta):

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def initialize_ocr():

    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def recognize_license_plate(image, ocr, x, y, w, h):

    plate_region = image[y:y+h, x:x+w]
    result = ocr.ocr(plate_region, cls=True)
    
    if result and result[0]:
        text = result[0][0][1][0]
        confidence = result[0][0][1][1]
        text = text.replace('I' , '1')
        text = text.replace('O' , '0')
        text = text.replace('$' , 'S')
        text = text.replace('.' , '-')
        return text, confidence
    return None, 0

def resize(image , weight , height):
    image = cv2.resize(image , (weight , height))
    return image 

def image_contrast(image , increase_multiple , increase_numerical_value):
    image = adjust_contrast(image , alpha = increase_multiple , beta = increase_numerical_value)
    return image

def mask_create(image , lower_h , lower_s , lower_v , upper_h , upper_s , upper_v):
    lower = np.array([lower_h , lower_s , lower_v])
    upper = np.array([upper_h , upper_s , upper_v])
    mask = cv2.inRange(image , lower , upper)
    return mask

def enhance_and_detect_with_ocr(image_path):

    ocr = initialize_ocr()
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("無法讀取圖像，請檢查路徑！")
        return
    
    image = resize(original_image, 1024, 800) #Resize image
    result_image = image.copy() 
    
    contrast = image_contrast(image , 1.3 , 11) # Contrast improve
    cv2.imshow("Adjusted Contrast", contrast)
    
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Image", hsv)

    white_mask = mask_create(hsv , 0 , 0 , 200 , 180 , 50 , 255)
    cv2.imshow("White Mask", white_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Step 4: Closed Mask", closed_mask)
    
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
    image_path = "pic/pic3.jpg"
    enhance_and_detect_with_ocr(image_path)
