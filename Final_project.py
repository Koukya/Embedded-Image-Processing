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
                cropped_image = image[y:y + h, x:x + w]
                cropped_path = f'cropped/crop_{x}_{y}.jpg'
                cv2.imwrite(cropped_path, cropped_image)
                detected_text = text
                max_confidence = confidence

    return detected_text, max_confidence


def process_video(video_path, output_dir, ocr):
    """處理單一影片並選出最長的車牌結果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_text = ""
    max_confidence = 0
    cropped_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # 每 30 幀提取一幀
            text, confidence = enhance_and_detect_with_ocr(frame, ocr)
            if text and len(text) > len(max_text):  # 更新最長的文字
                max_text = text
                max_confidence = confidence

            # 保存裁切圖片
            cropped_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]

    cap.release()
    return max_text, max_confidence, cropped_files


def display_images_sequentially(image_paths, value_texts):
    """以每秒間隔顯示裁切圖片"""
    window = tk.Tk()
    window.title("圖片與數值顯示")
    window.geometry("800x600")

    image_label = tk.Label(window)
    image_label.pack(pady=10)

    value_label = tk.Label(window, font=("Arial", 16), fg="blue")
    value_label.pack(pady=10)

    def update_image(index=0):
        if index < len(image_paths):
            img = Image.open(image_paths[index]).resize((320, 480))
            photo = ImageTk.PhotoImage(img)
            image_label.configure(image=photo)
            image_label.image = photo
            value_label.configure(text=f"車牌：{value_texts[index]}")
            window.after(1000, update_image, index + 1)

    update_image()
    window.mainloop()


def select_file():
    """選擇單一影片進行處理"""
    video_path = filedialog.askopenfilename(
        title="選擇影片",
        filetypes=[("影片檔案", "*.mp4;*.avi;*.mov")]
    )
    if video_path:
        ocr_engine = initialize_ocr()
        cropped_dir = "cropped"
        max_text, max_confidence, cropped_files = process_video(video_path, cropped_dir, ocr_engine)

        if cropped_files:
            display_images_sequentially(cropped_files, [max_text] * len(cropped_files))
        else:
            print("未檢測到任何車牌。")


if __name__ == "__main__":
    # 創建主視窗
    root = tk.Tk()
    root.title("單一影片車牌識別")
    root.geometry("400x200")

    # 添加按鈕
    select_button = tk.Button(root, text="選擇影片進行車牌識別", command=select_file, font=("Arial", 14))
    select_button.pack(expand=True)

    root.mainloop()
