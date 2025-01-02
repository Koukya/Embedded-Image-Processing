import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import os
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def initialize_ocr():
    """初始化 OCR 引擎"""
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def contains_letters_and_numbers(s):
    """判斷是否同時包含英文字符和數字"""
    has_letters = bool(re.search(r'[A-Z]', s))
    has_numbers = bool(re.search(r'\d', s))
    return has_letters and has_numbers

def recognize_license_plate(image, ocr, x, y, w, h):
    """識別車牌區域內的文字"""
    plate_region = image[y:y+h, x:x+w]
    result = ocr.ocr(plate_region, cls=True)
    
    if result and result[0]:
        text = result[0][0][1][0]
        confidence = result[0][0][1][1]
        # 修正常見錯誤字符
        text = text.replace('I', '1').replace('O', '0').replace('$', 'S').replace('.', '-')
        if contains_letters_and_numbers(text):
            return text, confidence
    return None, 0

def enhance_and_detect_with_ocr(image, ocr):
    """增強圖像並檢測車牌"""
    detected_text = ""
    max_confidence = 0
    
    contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=11)  # 調整對比
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    canny_image = cv2.Canny(closed_mask, 30, 200)
    
    contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        
        if 1000 < area < 200000 and 1.3 < aspect_ratio < 3.0 and h > 50:
            text, confidence = recognize_license_plate(image, ocr, x, y, w, h)
            if text and len(text) > len(detected_text):  # 更新最長的文字
                detected_text = text
                max_confidence = confidence
    
    return detected_text, max_confidence

def process_video(video_path, output_dir, ocr):
    """處理影片並選出最長的車牌結果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_text = ""
    max_confidence = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每 6 幀提取一幀
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # 處理圖片
            text, confidence = enhance_and_detect_with_ocr(frame, ocr)
            if text and len(text) > len(max_text):  # 更新最長的文字
                max_text = text
                max_confidence = confidence
    
    cap.release()
    print(f"影片處理完成！檢測到的最長車牌號碼為：{max_text} (置信度: {max_confidence:.2f})")
    return max_text, max_confidence

def display_image_and_text(image_path, value_text):
    # 創建主視窗
    window = tk.Tk()
    window.title("圖片與數值顯示")

    # 調整視窗大小
    window.geometry("800x600")

    # 加載圖片
    img = Image.open(image_path)
    img = img.resize((320 , 480))  # 調整圖片大小
    photo = ImageTk.PhotoImage(img)

    # 創建 Label 顯示圖片
    image_label = tk.Label(window, image=photo)
    image_label.pack(pady=10)

    # 顯示值的 Label
    value_label = tk.Label(window, text=f"車牌：{value_text}", font=("Arial", 16), fg="blue")
    value_label.pack(pady=10)

    # 啟動主迴圈
    window.mainloop()

if __name__ == "__main__":
    start_time = time.time()
    video_path = "pic/video1.mp4"
    output_dir = "video_frames"
    
    # 初始化 OCR 引擎
    ocr_engine = initialize_ocr()
    
    # 處理影片並取得最長結果
    longest_text, confidence = process_video(video_path, output_dir, ocr_engine)
    end_time = time.time()
    pass_time = end_time - start_time
    print(pass_time)
    
    image_path = "video_frames/frame_60.jpg"
    value_text = longest_text
    if image_path:
        display_image_and_text(image_path, value_text)

    
