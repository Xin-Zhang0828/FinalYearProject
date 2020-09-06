
import cv2

import numpy as np
from matplotlib import pyplot as plt

def blur():
    # 直接使用 cv2.blur（）函数进行均值滤波
    blur = cv2.blur(img, (5, 5))
    cv2.imwrite('./blur.jpg',blur)

    # 通过numpy构建均值滤波器模板，再使用cv2.filter2D()函数滤波
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('./dst.jpg',dst)
    imgName = [img, blur, dst]
    titleName = ['Original', 'Blurred', 'Averaged']

    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(imgName[i]), plt.title(titleName[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    cv2.imshow('original_img', img1)
    cv2.imshow("contrast_img", dst)

if __name__ == '__main__':
    img = cv2.imread("./360_combin/11.png", cv2.IMREAD_COLOR)
    contrast_img(img, 1.3, 3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()