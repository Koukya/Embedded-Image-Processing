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
    cv2.imwrite('pic/pic_canny.jpg',canny_pic)
    return canny_pic

img = cv2.imread('pic/pic1.jpg')
img_after = pre_image(img)



show_image('pic',img_after)
