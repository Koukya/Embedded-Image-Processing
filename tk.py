import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import os
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
    plate_region = image[y:y + h, x:x + w]
    result = ocr.ocr(plate_region, cls=True)

    if result and result[0]:
        text = result[0][0][1][0]
        confidence = result[0][0][1][1]
        text = text.replace('I', '1').replace('O', '0').replace('$', 'S').replace('.', '-')
        if contains_letters_and_numbers(text):
            return text, confidence
    return None, 0

def enhance_and_detect_with_ocr(image, ocr):
    """增強圖像並檢測車牌"""
    detected_text = ""
    max_confidence = 0
    best_plate_image = None

    contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=11)
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
            if text and confidence > max_confidence:
                max_confidence = confidence
                detected_text = text
                best_plate_image = image[y:y + h, x:x + w]

    return detected_text, max_confidence, best_plate_image

def process_video(video_path, ocr):
    """處理影片並選出最長的車牌結果"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_text = ""
    max_confidence = 0
    best_plate_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            text, confidence, plate_image = enhance_and_detect_with_ocr(frame, ocr)
            if text and confidence > max_confidence:
                max_text = text
                max_confidence = confidence
                best_plate_image = plate_image

    cap.release()
    return max_text, max_confidence, best_plate_image

def select_video():
    """讓使用者選擇影片並進行處理"""
    video_path = filedialog.askopenfilename(
        title="選擇影片",
        filetypes=[("影片文件", "*.mp4;*.avi;*.mov"), ("所有文件", "*.*")]
    )
    if video_path:
        ocr_engine = initialize_ocr()
        print(f"選擇的影片: {video_path}")
        detected_text, confidence, plate_image = process_video(video_path, ocr_engine)

        if confidence == 0:
            result_label.config(text="未成功辨識車牌")
            image_label.config(image="")
            image_label.image = None
        else:
            result_label.config(text=f"車牌號碼: {detected_text} (置信度: {confidence:.2f})")

            # 如果有車牌圖片，顯示於界面
            if plate_image is not None:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(plate_image)
                pil_image = pil_image.resize((320, 120))  # 調整顯示大小
                photo = ImageTk.PhotoImage(pil_image)
                image_label.config(image=photo)
                image_label.image = photo

if __name__ == "__main__":
    # 建立主視窗
    root = tk.Tk()
    root.title("車牌識別系統")
    root.geometry("600x400")

    # 按鈕讓使用者選擇影片
    select_button = tk.Button(root, text="選擇影片", command=select_video, font=("Arial", 14))
    select_button.pack(pady=20)

    # 顯示結果的 Label
    result_label = tk.Label(root, text="車牌號碼將顯示於此", font=("Arial", 12), fg="blue")
    result_label.pack(pady=20)

    # 顯示車牌圖片的 Label
    image_label = tk.Label(root)
    image_label.pack(pady=20)

    # 啟動主迴圈
    root.mainloop()
