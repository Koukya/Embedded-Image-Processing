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

# 六、成果展示
# 七、結論
本次實作成功結合了OpenCV與OCR的銜接，透過一些前處理調整原圖後，選出可能是車牌的輪廓，再利用OCR來辨識輪廓內的文字，最後將所得到的結果禁行後處理(篩選規則內的車牌及多個車牌號碼裡最可能是的那一個)，並將結果顯示到Tkinter的視窗上 
# 八、參考資料
https://github.com/LonelyCaesar/OpenCV-license-plate-recognition  
https://blog.csdn.net/lsb2002/article/details/134415492  
https://hackmd.io/@CynthiaChuang/Taiwan-License-Plate-Rules-for-LPR  
https://hackmd.io/@kenchick/SyxT90UPd  
https://medium.com/jia-hong/%E5%9F%BA%E6%96%BCopencv%E4%B9%8B%E8%BB%8A%E7%89%8C%E8%BE%A8%E8%AD%98-b14ca20b1803  
