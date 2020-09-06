import os
import shutil
import time

import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import cv2

def Get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #print(root)  # 当前目录路径
        #print(dirs)  # 当前路径下所有子目录
        #print(files)  # 当前路径下所有非目录子文件
        pass
    #print(files)
    return root,dirs,files

def crop_img_SIFT(inputpath, path_split):

    if not os.path.exists(inputpath):
        os.mkdir(inputpath)
    if not os.path.exists(path_split):
        os.mkdir(path_split)
    root,dirs,piclist=Get_file_name(inputpath)

    temppath= './test/sift_temp/'
    # print(temppath)
    # exit(1)
    for file in piclist:
        filename=file.split('.')[0]
        if not os.path.exists(path_split +'/'+filename):
            os.mkdir(path_split +'/'+filename)
        black=100
        #扩展图片
        img = cv2.imread(inputpath+'/'+file)
        a = cv2.copyMakeBorder(img, 0, 100, 0, black, cv2.BORDER_REPLICATE)
        cv2.imwrite(temppath + file, a)
        time.sleep(1)
        img = Image.open(temppath+file)
        # print(img)
        # exit(1)
        for i in range(8):
            if i==0:
                a=0
                b=0
                c=580
                d=580
            elif i==1:
                a = 480
                b = 0
                c = 1060
                d = 580
            elif i==2:
                a = 960
                b = 0
                c = 1540
                d = 580
            elif i==3:
                a = 1440
                b = 0
                c = 2020
                d = 580
            elif i==4:
                a = 0
                b = 480
                c = 580
                d = 1060
            elif i==5:
                a = 480
                b = 480
                c = 1060
                d = 1060
            elif i==6:
                a = 960
                b = 480
                c = 1540
                d = 1060
            elif i==7:
                a = 1440
                b = 480
                c = 2020
                d = 1060
           
            #print((a,b,c,d))
            cropimg = img.crop((a,b,c,d))
            cropimg.save(path_split +  filename+'_'+str(i) + '.png')


def crop_img_SIFT_360(inputpath, path_split):
    if not os.path.exists(inputpath):
        os.mkdir(inputpath)
    if not os.path.exists(path_split):
        os.mkdir(path_split)
    root, dirs, piclist = Get_file_name(inputpath)

    temppath = './test/sift_temp/'
    # print(temppath)
    # exit(1)
    for file in piclist:
        filename = file.split('.')[0]
        if not os.path.exists(path_split + '/' + filename):
            os.mkdir(path_split + '/' + filename)

        #合并两三张图片，左 上 右  下  左右各多200像素
        img1 = Image.open(inputpath + '/' + file)
        target = Image.new('RGB', (5760, 960))
        target.paste(img1, (0,0,1920,960))
        target.paste(img1, (1920, 0, 3840, 960))
        target.paste(img1, (3840, 0, 5760, 960))
        #裁剪
        cropimg1 = target.crop((1720,0,4040,960 ))
        cropimg1=np.array(cropimg1)
        #得到 2320*960   (2320+50)/3=790 交错25像素切
        # 扩展图片
        #img = cv2.imread(inputpath + '/' + file)
        black = 113
        #a = cv2.copyMakeBorder(cropimg1, 0, 0, 0, black, cv2.BORDER_REPLICATE) #得到 2433*960
        #a.save(temppath + file)
        img3 = cv2.cvtColor(cropimg1, cv2.COLOR_BGR2RGB)
        cv2.imwrite(temppath + file, img3)
        time.sleep(1)
        img = Image.open(temppath + file)
        # print(img)
        #exit(1)
        #切割部分有100重合位置，每张840
        # for i in range(3):
        #     if i == 0:
        #         a = 0
        #         b = 0
        #         c = 790
        #         d = 960
        #     elif i == 1:
        #         a = 765
        #         b = 0
        #         c = 1555
        #         d = 960
        #     elif i == 2:
        #         a = 1530
        #         b = 0
        #         c = 2320
        #         d = 960
        for i in range(3):
            if i == 0:
                a = 0
                b = 0
                c = 840
                d = 960
            elif i == 1:
                a = 740
                b = 0
                c = 1580
                d = 960
            elif i == 2:
                a = 1480
                b = 0
                c = 2320
                d = 960
            # print((a,b,c,d))
            cropimg = img.crop((a, b, c, d))
            cropimg = np.array(cropimg)
            a = cv2.copyMakeBorder(cropimg, 200, 200, 200, 200, cv2.BORDER_REPLICATE)
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_split + filename + '_' + str(i) + '.png', a)
            #a.save(path_split + filename + '_' + str(i) + '.png')

def combin_All(inputpath,names,output):

    namelist = []
    for pic in names:
        name = pic.split('.')[0].split('_')[0]
        if name not in namelist:
            namelist.append(name)
    tempP='./test/sift_temp/'
    #清空缓存文件夹
    if not os.path.exists(tempP):
        os.mkdir(tempP)
    else:
        shutil.rmtree(tempP)
        os.mkdir(tempP)
    for name in namelist:
        #target = Image.new('RGB', (1920, 1440))
        tempname1=tempP+name+'_temp1.png'
        tempname2 = tempP + name + '_temp2.png'
        tempname1_2=tempP + name + '_temp1_2.png'
        pic0=inputpath+name+'_0.png'
        pic1 = inputpath + name + '_1.png'
        temp=test(pic0, pic1,480)
        cv2.imwrite(tempname1, temp)

        pic2 = inputpath + name + '_2.png'
        pic3 = inputpath + name + '_3.png'
        temp = test(pic2, pic3,480)
        cv2.imwrite(tempname2, temp)

        temp = test(tempname1, tempname2,960)
        #上下裁剪200，右边裁剪20
        temp = temp[200:880, 0:2020]  # 裁剪坐标为[y0:y1, x0:x1]
        cv2.imwrite(tempname1_2, temp)

        #下半边

        tempname1 = tempP + name + '_temp3.png'
        tempname2 = tempP + name + '_temp4.png'
        tempname3_4 = tempP + name + '_temp3_4.png'
        pic0 = inputpath + name + '_4.png'
        pic1 = inputpath + name + '_5.png'
        temp = test(pic0, pic1, 480)
        cv2.imwrite(tempname1, temp)

        pic2 = inputpath + name + '_6.png'
        pic3 = inputpath + name + '_7.png'
        temp = test(pic2, pic3, 480)
        cv2.imwrite(tempname2, temp)

        temp = test(tempname1, tempname2, 960)
        # 上下裁剪200，右边裁剪20
        temp = temp[200:880, 0:2020]  # 裁剪坐标为[y0:y1, x0:x1]
        cv2.imwrite(tempname3_4, temp)

        #上下合并
        temp = test1(tempname1_2, tempname3_4, 680)
        temp = RotateClockWise90(temp)
        temp = temp[0:960, 100:2120]  # 裁剪坐标为[y0:y1, x0:x1]
        cv2.imwrite(output+name+ '.png', temp)
        #exit(1)

def combin_All_new360(inputpath,names,output):

    namelist = []
    for pic in names:
        name = pic.split('.')[0].split('_')[0]
        if name not in namelist:
            namelist.append(name)
    tempP='./test/sift_temp/'
    #清空缓存文件夹
    if not os.path.exists(tempP):
        os.mkdir(tempP)
    else:
        shutil.rmtree(tempP)
        os.mkdir(tempP)
    #裁剪风格化后的多余边界

    for name in names:
        im = Image.open(inputpath+name)
        x = 200
        y = 200
        #w = 790
        w = 840
        h = 960
        region = im.crop((x, y, x + w, y + h))
        region.save(inputpath+name)
    #exit(1)
    for name in namelist:
        #target1 = Image.new('RGB', (1920, 960))
        tempname0_1=tempP+name+'_temp0_1.png'
        pic0=inputpath+name+'_0.png'
        pic1 = inputpath + name + '_1.png'
        pic2 = inputpath + name + '_2.png'
        #temp=test(pic0, pic1,790)
        #cv2.imwrite(tempname0_1, temp)

        #test
        target = Image.new('RGB', (1920, 960))
        img0 = Image.open(pic0)
        img1 = Image.open(pic1)
        img2 = Image.open(pic2)
        img0 = img0.crop((200, 0, 790, 960)) #590
        img1 = img1.crop((50, 0, 790, 960))#740
        img2 = img2.crop((50, 0, 640, 960))#590
        target.paste(img0, (0, 0, 590, 960))
        target.paste(img1, (590, 0, 1330, 960))
        target.paste(img2, (1330, 0, 1920, 960))

        # img0 = img0.crop((200, 0, 765, 960))  # 565
        # img1 = img1.crop((0, 0, 765, 960))  # 765
        # img2 = img2.crop((0, 0, 590, 960))  # 590
        # target.paste(img0, (0, 0, 565, 960))
        # target.paste(img1, (565, 0, 1330, 960))
        # target.paste(img2, (1330, 0, 1920, 960))
        target.save(output+name+ '.png')
        #over

        # tempall = test(tempname0_1, pic2,790)
        # #cv2.imwrite(tempname2, temp)
        #
        # #上下裁剪200，右边裁剪20
        # #cv2.imwrite(tempname1_2, tempall)
        # tempall = tempall[0:960, 200:2120]  # 裁剪坐标为[y0:y1, x0:x1]
        # cv2.imwrite(output+name+ '.png', tempall)
        #exit(1)

# 顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img


# 逆时针旋转90度
def RotateAntiClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip( trans_img, 0 )
    return new_img

def test1(pic1,pic2,right):
    top, bot, left = 100, 100, 0
    img1 = cv.imread(pic1)
    img2 = cv.imread(pic2)
    img1=RotateAntiClockWise90(img1)
    img2 = RotateAntiClockWise90(img2)

    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))

    # exit(1)
    # img1gray = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
    # img2gray = cv.cvtColor(testImg, cv.COLOR_RGB2GRAY)
    img1gray = srcImg
    img2gray = testImg
    sift = cv.xfeatures2d_SIFT().create()
    # sift =  cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    rows, cols = srcImg.shape[:2]
    # print(rows, cols)
    # exit(1)
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv.WARP_INVERSE_MAP)

        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break

        # res = np.zeros([rows, cols, 3], np.uint8)
        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

        # opencv is bgr, matplotlib is rgb
        #res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        res = cv.cvtColor(res, cv.COLOR_RGB2BGR)

        return res

def test(pic1,pic2,right):
    top, bot, left = 100, 100, 0
    img1 = cv.imread(pic1)
    img2 = cv.imread(pic2)

    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))

    # exit(1)
    # img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    # img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
    img1gray = srcImg
    img2gray = testImg
    sift = cv.xfeatures2d_SIFT().create()
    # sift =  cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    rows, cols = srcImg.shape[:2]
    # print(rows, cols)
    # exit(1)
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv.WARP_INVERSE_MAP)

        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break

        #res = np.zeros([rows, cols, 3], np.uint8)
        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

        # opencv is bgr, matplotlib is rgb
        # res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        res = cv.cvtColor(res, cv.COLOR_RGB2BGR)
        #cropped = res[0:128, 0:512]
        # show the result
        # plt.figure()
        # plt.imshow(res)
        # plt.show()
        return res
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

if __name__ == '__main__':
    pic1='../test/360/'
    pic2='../test/sift_crop/'
    #test(pic1,pic2)
    #crop_img_SIFT(pic1, pic2)
    inputpath ='./test/sift_fenggehua/'
    output = './test/sift_out/'
    #combin_ALl(inputpath, output)