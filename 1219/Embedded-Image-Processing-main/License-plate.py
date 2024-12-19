import cv2
import numpy
import matplotlib.pyplot as plt


def show_image(name , image):
    cv2.imshow(name,image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def pre_image(img):
    img = cv2.resize(img, (1024 , 800))
    gray_pic = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    bil_pic = cv2.bilateralFilter(gray_pic, 13 , 15 , 15)
    canny_pic = cv2.Canny(bil_pic , 30 , 200)
    return canny_pic

img = cv2.imread('pic/pic1.jpg')
img_after = pre_image(img)
contours , _ = cv2.findContours(img_after.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours , key = cv2.contourArea , reverse = True)[:10]
screenCnt = None

for c in contours:
    if cv2.contourArea(c) > 1024 * 768 * 0.05:
        continue

    peri = cv2.arcLength(c , True)

    approx = cv2.approxPolyDP(c , 0.018 * peri , True)

    if len(approx) == 4:
        crop_image = img[approx[3][0][1]:approx[0][0][1] , approx[3][0][0]:approx[2][0][0]]
        show_image('crop',crop_image)
        screenCnt = approx
        breakpoint
if screenCnt is not None:
    cv2.drawContours(img, [screenCnt] , -1 , (0, 0 , 255) ,3)
    show_image('contour',img)


show_image('pic',img)
