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
# 五、程式說明
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
# 六、成果展示
### 介面
![image](https://github.com/user-attachments/assets/e8d27f9f-9b3d-4ef4-bca3-be7445ef46a5)  
### 辨識成功(模擬入場)
![image](https://github.com/user-attachments/assets/744f19c6-91d2-4677-9810-4a8572095595)  
### 無法辨識
![image](https://github.com/user-attachments/assets/6f0bb705-ce20-4728-99b8-4578b1247917)  
### 重複時取消(模擬離場)
![image](https://github.com/user-attachments/assets/94b2dc7b-0741-4041-a829-0d579142663e)

# 七、結論
本次實作成功結合了OpenCV與OCR的銜接，透過一些前處理調整原圖後，選出可能是車牌的輪廓，再利用OCR來辨識輪廓內的文字，最後將所得到的結果進行後處理(篩選規則內的車牌及多個車牌號碼裡最可能是的那一個)，並將結果顯示到Tkinter的視窗上 
# 八、參考資料
https://github.com/LonelyCaesar/OpenCV-license-plate-recognition  
https://blog.csdn.net/lsb2002/article/details/134415492  
https://hackmd.io/@CynthiaChuang/Taiwan-License-Plate-Rules-for-LPR  
https://hackmd.io/@kenchick/SyxT90UPd  
https://medium.com/jia-hong/%E5%9F%BA%E6%96%BCopencv%E4%B9%8B%E8%BB%8A%E7%89%8C%E8%BE%A8%E8%AD%98-b14ca20b1803  
