import cv2
import numpy as np
from paddleocr import PaddleOCR

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def preprocess_image(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    cv2.imshow("Original Image", image)

    # 調整大小
    image = cv2.resize(image, (1024, 800))

    # 灰階處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)

    # 高斯模糊 + 邊緣檢測
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 110)
    cv2.imshow("Edges Detected", edges)

    # 尋找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 過濾輪廓：根據面積和比例篩選候選區域
    candidates = []
    for cnt in contours:
        # 計算輪廓的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        if 2 < aspect_ratio < 6 and 1000 < area < 50000:  # 假設車牌比例範圍與大小
            candidates.append((x, y, w, h))

    # 排序候選區域，選取前 10 大區域
    candidates = sorted(candidates, key=lambda item: item[2] * item[3], reverse=True)[:3]
    return image, candidates

def recognize_plate_number(image, candidates):
    if not candidates:
        print("無法識別車牌：未檢測到候選區域")
        return None

    results = []
    for x, y, w, h in candidates:
        # 提取區域
        plate_image = image[y:y+h, x:x+w]
        cv2.imshow(f"Candidate Region ({x}, {y})", plate_image)

        # OCR 辨識
        result = ocr.ocr(plate_image)
        if result and result[0]:
            plate_number = result[0][0][1][0]
            confidence = result[0][0][1][1]
            results.append((plate_number, confidence))
            print(f"候選區域 {x}, {y}, OCR 識別結果: {plate_number}, 置信度: {confidence}")

    # 選擇置信度最高的結果
    if results:
        best_result = max(results, key=lambda x: x[1])
        print(f"最終選擇車牌號碼: {best_result[0]}, 置信度: {best_result[1]}")
        return best_result[0]

    print("未能識別車牌號碼")
    return None

# 主程式
image_path = r"D:\Microsoft VS Code\python\pic1.jpg"
image, candidates = preprocess_image(image_path)
plate_number = recognize_plate_number(image, candidates)

# 輸出結果
if plate_number:
    print("車牌號碼:", plate_number)
else:
    print("未能識別車牌號碼")

cv2.waitKey(0)
cv2.destroyAllWindows()
