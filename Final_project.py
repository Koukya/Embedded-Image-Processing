import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def initialize_ocr():

    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def contains_letters_and_numbers(s):

    has_letters = bool(re.search(r'[A-Z]', s))
    has_numbers = bool(re.search(r'\d', s))
    return has_letters and has_numbers


def recognize_license_plate(image, ocr, x, y, w, h):

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

    detected_text = ""
    max_confidence = 0
    best_plate_image = None

    image = cv2.resize(image, (1080, 1080))
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
        if frame_count % 10 == 0:  
            text, confidence, plate_image = enhance_and_detect_with_ocr(frame, ocr)
            if text and confidence > max_confidence:
                max_text = text
                max_confidence = confidence
                best_plate_image = plate_image
        if (len(max_text) == 7):
            max_text = max_text[:3] + '-' + max_text[3:]
            replace_text = max_text[4:]
            replace_text = replace_text.replace('Z' , '2')
            max_text = max_text[:4] + replace_text
    cap.release()
    return max_text, max_confidence, best_plate_image


def select_folder():

    folder_path = filedialog.askdirectory(title="選擇包含影片的資料夾")
    if folder_path:
        process_folder(folder_path)


def process_folder(folder_path):

    global video_files, current_index
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    current_index = 0
    if video_files:
        process_next_video()
    else:
        result_label.config(text="未找到任何影片")


def process_next_video():

    global current_index, video_files, ocr_engine

    if current_index < len(video_files):
        video_path = video_files[current_index]
        detected_text, confidence, plate_image = process_video(video_path, ocr_engine)

        if confidence == 0:
            result_label.config(text=f"影片: {os.path.basename(video_path)}\n未成功辨識車牌")
            image_label.config(image="")
            image_label.image = None
        else:
            result_label.config(text=f"影片: {os.path.basename(video_path)}\n車牌號碼: {detected_text} (置信度: {confidence:.2f})")

            if plate_image is not None:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(plate_image)
                pil_image = pil_image.resize((320, 120))  
                photo = ImageTk.PhotoImage(pil_image)
                image_label.config(image=photo)
                image_label.image = photo

            existing_plates = plate_listbox.get(0, tk.END)
            if detected_text in existing_plates:
                index = existing_plates.index(detected_text)
                plate_listbox.delete(index)  
            else:
                plate_listbox.insert(tk.END, detected_text)

        current_index += 1
        root.after(1000, process_next_video) 
    else:
        result_label.config(text="處理完成！")


if __name__ == "__main__":
    # 初始化 OCR 引擎
    ocr_engine = initialize_ocr()

    # 建立主視窗
    root = tk.Tk()
    root.title("車牌識別系統")
    root.geometry("800x400")

    # 左側區域
    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # 按鈕讓使用者選擇資料夾
    select_button = tk.Button(left_frame, text="選擇資料夾", command=select_folder, font=("Arial", 14))
    select_button.pack(pady=20)

    # 顯示結果的 Label
    result_label = tk.Label(left_frame, text="車牌號碼將顯示於此", font=("Arial", 12), fg="blue")
    result_label.pack(pady=20)

    # 顯示車牌圖片的 Label
    image_label = tk.Label(left_frame)
    image_label.pack(pady=20)

    # 右側區域
    right_frame = tk.Frame(root)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    # 車牌列表 Label
    list_label = tk.Label(right_frame, text="已偵測車牌", font=("Arial", 14))
    list_label.pack(pady=10)

    # 車牌列表框
    plate_listbox = tk.Listbox(right_frame, font=("Arial", 12), height=15, width=20)
    plate_listbox.pack(pady=10)

    # 啟動主迴圈
    root.mainloop()
