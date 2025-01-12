# Embedded-Image-Processing
# 一、說明
本次專題主要透過OpenCV加上OCR(光學字元辨識)來實現車牌辨識功能
# 二、需求
### 功能
1.辨識圖片中的車牌號碼  
2.將入場的車牌紀錄及離場的車牌刪除  
3.能隨時檢視停車場內的車牌  
### 畫面大小
#### 輸入
720 * 1080 pixels  
29.99FPS  
#### 輸出
擷取後車牌(720 X 1280 pixels)  
### 效能
每次一張圖片，30FPS取五張圖片
### 介面
1.顯示目前車牌圖片及號碼  
2.顯示目前車輛總數及剩餘位置
### 限制
1.一次一張車牌  
2.中華民國車牌  
3.正面車牌(不用正前方)
# 三、Breakdown
![image](https://github.com/user-attachments/assets/060c3bfd-656b-432b-90dc-a8296daf4b66)
# 四、流程圖
![image](https://github.com/user-attachments/assets/1ebddb0c-4fdf-4f08-82da-00ca9bba2f6c)
# 五、API
![image](https://github.com/user-attachments/assets/ad39cd4a-fbfa-404b-9dd1-dddd56cd188b)  
![image](https://github.com/user-attachments/assets/e212a19a-d08c-4218-9c6c-b09e8b72b7a4)  
![image](https://github.com/user-attachments/assets/34f3e869-9c4b-4a89-8e24-18553836d4c8)  
![image](https://github.com/user-attachments/assets/217b9e34-9b40-420d-8bb3-1bebf36be269)  
![image](https://github.com/user-attachments/assets/81c4bcd5-0409-4c94-b125-5b92ed1eec9e)  
![image](https://github.com/user-attachments/assets/f8e6d3ef-f084-4c21-bd3b-908abc6bfb73)
![image](https://github.com/user-attachments/assets/773e04e5-e4f8-4281-a21e-257cb64b0e96)
![image](https://github.com/user-attachments/assets/089bb551-acdd-4c41-99f9-104c09df5da6)
# 六、程式說明
### 影片處理
創立資料夾用來存放提取的圖片，並取名為video_frames
```
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

output_dir = "video_frames"
```
圖片提取方式
```
frame_count = 0

frame_count += 1
        if frame_count % 6 == 0:  # 每 6 幀提取一幀
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
```
<video src = "https://github.com/Koukya/Embedded-Image-Processing/blob/main/pic/video1.mp4"></video>
圖片處理，包含調整對比、HSV找白色區域、應用形態學以及Canny
```
contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=11)  # 調整對比
hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)
white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
closed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
canny_image = cv2.Canny(closed_mask, 30, 200)
```
區域判斷並記錄區域內最長的文字(提高辨識成功率)
```
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
```
辨識文字方法以及針對車牌邏輯做出文字修正
```
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

def contains_letters_and_numbers(s):
    """判斷是否同時包含英文字符和數字"""
    has_letters = bool(re.search(r'[A-Z]', s))
    has_numbers = bool(re.search(r'\d', s))
    return has_letters and has_numbers
```
創建Tkinter主視窗並給定視窗名稱與大小  
```
root = tk.Tk()
root.title("車牌識別系統")
root.geometry("800x400")
```
創建左側區域給定區域定位  
```
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)
```
在左側區域創建用於選擇資料夾的按鈕  
```
select_button = tk.Button(left_frame, text="選擇資料夾", command=select_folder, font=("Arial", 14))
select_button.pack(pady=20)
```
在左側區域創建文字區域用於顯示偵測後結果  
```
result_label = tk.Label(left_frame, text="車牌號碼將顯示於此", font=("Arial", 12), fg="blue")
result_label.pack(pady=20)
```
在左側區域創建區域用於顯示擷取後的車牌  
```
image_label = tk.Label(left_frame)
image_label.pack(pady=20)
```
創建右側區域給定區域定位  
```
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
```
在右側區域創建區域用於顯示表格上方的文字  
```
list_label = tk.Label(right_frame, text="已偵測車牌", font=("Arial", 14))
list_label.pack(pady=10)
```
在右側區域創建表格用於顯示場內車牌號碼  
```
plate_listbox = tk.Listbox(right_frame, font=("Arial", 12), height=15, width=20)
plate_listbox.pack(pady=10)
```
啟動主迴圈  
```
root.mainloop()
```
# 七、成果展示
### 介面
![image](https://github.com/user-attachments/assets/e8d27f9f-9b3d-4ef4-bca3-be7445ef46a5)  
### 辨識成功(模擬入場)
![image](https://github.com/user-attachments/assets/744f19c6-91d2-4677-9810-4a8572095595)  
### 無法辨識
![image](https://github.com/user-attachments/assets/6f0bb705-ce20-4728-99b8-4578b1247917)  
### 重複時取消(模擬離場)
![image](https://github.com/user-attachments/assets/94b2dc7b-0741-4041-a829-0d579142663e)

# 八、結論
本次實作成功結合了OpenCV與OCR的銜接，透過一些前處理調整原圖後，選出可能是車牌的輪廓，再利用OCR來辨識輪廓內的文字，最後將所得到的結果進行後處理(篩選規則內的車牌及多個車牌號碼裡最可能是的那一個)，並將結果顯示到Tkinter的視窗上 
# 九、參考資料
https://github.com/LonelyCaesar/OpenCV-license-plate-recognition  
https://blog.csdn.net/lsb2002/article/details/134415492  
https://hackmd.io/@CynthiaChuang/Taiwan-License-Plate-Rules-for-LPR  
https://hackmd.io/@kenchick/SyxT90UPd  
https://medium.com/jia-hong/%E5%9F%BA%E6%96%BCopencv%E4%B9%8B%E8%BB%8A%E7%89%8C%E8%BE%A8%E8%AD%98-b14ca20b1803  
