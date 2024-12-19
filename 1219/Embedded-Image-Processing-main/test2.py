import cv2
import numpy as np

def adjust_contrast(image, alpha=1.3, beta=1000):
    """
    調整圖片的對比度與亮度
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_and_detect_with_closing(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("無法讀取圖片，請確認路徑正確！")
        return
    
    image = cv2.resize(image, (1024, 800))
    cv2.imshow("Original Image", image)

    # Step 1: 提高整體對比度
    image_contrast = adjust_contrast(image, alpha=1.3, beta=10)
    cv2.imshow("Contrast Adjusted Image", image_contrast)
    cv2.imwrite('contrast.jpg' , image_contrast)

    # Step 2: 轉換成 HSV 色彩空間
    hsv = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Image", hsv)
    cv2.imwrite('HSV.jpg' , hsv) 


    # Step 3: 偵測白色區域的 HSV 範圍
    lower_white = np.array([0, 0, 200])  # H:0-180, S:0-50, V:221-255
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow("white",white_mask)
    cv2.imwrite('white.jpg',white_mask)

    # Step 4: 應用形態學閉操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))  # 調整大小影響處理效果
    closed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed Mask", closed_mask)
    cv2.imwrite('mask.jpg',closed_mask)

    # 增強白色區域
    enhanced_white = cv2.addWeighted(image_contrast, 1.5, image_contrast, 0, 30)
    enhanced_image = cv2.bitwise_and(enhanced_white, enhanced_white, mask=closed_mask)

    # 弱化非白色區域
    inverted_mask = cv2.bitwise_not(closed_mask)
    weakened_non_white = cv2.addWeighted(image, 0.5, image, 0, -50)
    weakened_image = cv2.bitwise_and(weakened_non_white, weakened_non_white, mask=inverted_mask)

    # 合成結果
    final_image = cv2.add(enhanced_image, weakened_image)
    cv2.imshow("Enhanced White Regions with Closing", final_image)
    cv2.imwrite('final.jpg' , final_image)

    # Step 5: 使用遮罩找車牌候選區域
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        # 車牌篩選條件
        if 1000 < area < 200000 and 1.3 < aspect_ratio < 6.0 and h > 50:
            candidates.append((x, y, w, h))
            cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detected License Plates", final_image)
    cv2.imwrite('final_1.jpg',final_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程式
image_path = "pic/pic1.jpg"  # 圖片路徑
enhance_and_detect_with_closing(image_path)
