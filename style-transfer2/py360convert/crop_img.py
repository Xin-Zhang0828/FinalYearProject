#!/usr/bin/env python3
from PIL import Image
import os
import shutil

import cv2
from matplotlib import pyplot as plt

import combinimg
import filter as ft


def Get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #print(root)  # 当前目录路径
        #print(dirs)  # 当前路径下所有子目录
        #print(files)  # 当前路径下所有非目录子文件
        pass
    #print(files)
    return root,dirs,files

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def add_black_bound(inpath,temppath):
    dirs= get_immediate_subdirectories(inpath)
    black=200
    for dir in dirs:
        # if not os.path.exists(temppath+dir):
        #     os.mkdir(temppath+dir)
        root,d,filenames=Get_file_name(inpath+dir)
        for file in filenames:
            img = Image.open(inpath+dir+'/'+file)
            new_pic = Image.new('RGBA', (480 + black, 480 + black),color=(0,255,0))
            new_pic.paste(img, (black//2, black//2))
            new_pic.save(temppath+file)

def add_copy_bound(inpath,temppath):
    dirs= get_immediate_subdirectories(inpath)
    black=100
    for dir in dirs:
        # if not os.path.exists(temppath+dir):
        #     os.mkdir(temppath+dir)
        root,d,filenames=Get_file_name(inpath+dir)
        for file in filenames:
            img = cv2.imread(inpath+dir+'/'+file)
            a = cv2.copyMakeBorder(img, black, black, black, black, cv2.BORDER_REPLICATE)
            #a.save(temppath + file)
            cv2.imwrite(temppath + file, a)

def add_copy_bound_cube(inpath,temppath):
    black1=100
    root,d,filenames=Get_file_name(inpath)
    for file in filenames:
        img = cv2.imread(inpath+file)
        a = cv2.copyMakeBorder(img, black1, black1, black1, black1, cv2.BORDER_REPLICATE)
        #a.save(temppath + file)
        cv2.imwrite(temppath + file, a)

def del_black_bound(inpath,savepath):
    root,d,filenames=Get_file_name(inpath)
    for file in filenames:
        im = Image.open(inpath+file)
        x = 100
        y = 100
        w = 480
        h = 480
        region = im.crop((x, y, x + w, y + h))
        region.save(savepath+file)

def del_bound_cube(noblackpath,blackpath,savepath):
    copyf(noblackpath, savepath)

    root,d,filenames=Get_file_name(savepath)
    for file in filenames:
        im = Image.open(savepath+file)
        x = 100
        y = 100
        w = 480
        h = 480
        region = im.crop((x, y, x + w, y + h))
        region.save(savepath+file)
    copyf(blackpath, savepath)

def copyf(path, out):
    filename = os.listdir(path)
    for i in filename:
        pic = os.path.join(path, i)
        # shutil.copy(newFilePath+'/'+i,filePath+'/'+i)
        shutil.copy(pic, out + '/' + i)



def crop_img(imgpath, filename):
    path_black = './test/black/'
    black_list = [0,2,3,8,10,11]
    path_notblack = './test/notblack/'
    if not os.path.exists(path_black):
        os.mkdir(path_black)
    if not os.path.exists(path_notblack):
        os.mkdir(path_notblack)

    img = Image.open(imgpath)
    size = img.size
    crop_size = 480
    #print(size)
    shang = 0
    left = 0
    index = 0
    for i in range(12):
        if i != 0 and i % 4 == 0:
            shang += 1
            left = 0
        a = crop_size * left
        b = crop_size * shang
        c = crop_size * (left + 1)
        d = crop_size * (shang +1)
        print((a,b,c,d))
        cropimg = img.crop((a,b,c,d))
        if index not in black_list:
            cropimg.save(path_notblack + filename + '_' + str(index) + '.png')
        else:
            cropimg.save(path_black + filename + '_' + str(index) + '.png')

        index += 1
        left += 1




def crop_img_360(inputpath, path_split):

    if not os.path.exists(inputpath):
        os.mkdir(inputpath)
    if not os.path.exists(path_split):
        os.mkdir(path_split)
    root,dirs,piclist=Get_file_name(inputpath)

    for file in piclist:
        filename=file.split('.')[0]
        if not os.path.exists(path_split +'/'+filename):
            os.mkdir(path_split +'/'+filename)

        img = Image.open(inputpath+'/'+file)

        #crop_size = 480
        crop_size = 480
        black_size = 60
        shang = 0
        left = 0
        index = 0
        for i in range(8):
            if i != 0 and i % 4 == 0:
                shang += 1
                left = 0
            a = crop_size * left
            b = crop_size * shang
            c = crop_size * (left + 1)
            d = crop_size * (shang + 1)

            cropimg = img.crop((a,b,c,d))
            cropimg.save(path_split +'/'+filename +'/'+  filename+'_'+str(index) + '.png')
            left += 1
            index+=1



def joint_img(path,savepath):

    target = Image.new('RGB',(1920,1440))
    crop_size = 480
    shang = 0
    left = 0
    index = 0
    for i in range(12):
        if i != 0 and i % 4 == 0:
            shang += 1
            left = 0
        a = crop_size * left
        b = crop_size * shang
        c = crop_size * (left + 1)
        d = crop_size * (shang +1)

        #print((a,b,c,d))

        img = Image.open(path +'/' + str(i) + '.png')
        target.paste(img,(a,b,c,d))

        index += 1
        left += 1
    target.save(savepath)



def joint_360_img(inputpath,savepath):
    dirs = get_immediate_subdirectories(inputpath)
    for dir in dirs:
        target = Image.new('RGB',(1920,960))
        crop_size = 480
        shang = 0
        left = 0
        index = 0
        for i in range(8):
            if i != 0 and i % 4 == 0:
                shang += 1
                left = 0
            a = crop_size * left
            b = crop_size * shang
            c = crop_size * (left + 1)
            d = crop_size * (shang +1)
            #print((a,b,c,d))

            img = Image.open(inputpath+'/'+dir +'/' + str(index) + '.png')
            target.paste(img,(a,b,c,d))

            index += 1
            left += 1
        target.save(savepath+'/'+dir+'.png')

def Get_filename_360_split(inputpath):
    root, dirs, piclist = Get_file_name(inputpath)
    namelist=[]
    for pic in piclist:
        name=pic.split('.')[0].split('_')[0]
        if name not in namelist:
            namelist.append(name)
    return namelist

def joint_360_img_allin(inputpath,savepath):
    names = Get_filename_360_split(inputpath)
    for name in names:
        target = Image.new('RGB',(1920,960))
        crop_size = 480
        shang = 0
        left = 0
        index = 0
        for i in range(8):
            if i != 0 and i % 4 == 0:
                shang += 1
                left = 0
            a = crop_size * left
            b = crop_size * shang
            c = crop_size * (left + 1)
            d = crop_size * (shang +1)
            #print((a,b,c,d))
            img = Image.open(inputpath+'/'+name +'_' + str(index) + '.png')
            target.paste(img,(a,b,c,d))

            index += 1
            left += 1
        target.save(savepath+'/'+name+'.png')


def joint_cube_img_allin(inputpath,savepath):
    names = Get_filename_360_split(inputpath)
    for name in names:
        target = Image.new('RGB',(1920,1440))
        crop_size = 480
        shang = 0
        left = 0
        index = 0
        for i in range(12):
            if i != 0 and i % 4 == 0:
                shang += 1
                left = 0
            a = crop_size * left
            b = crop_size * shang
            c = crop_size * (left + 1)
            d = crop_size * (shang +1)
            #print((a,b,c,d))
            img = Image.open(inputpath+'/'+name +'_' + str(index) + '.png')
            target.paste(img,(a,b,c,d))
            index += 1
            left += 1
        target.save(savepath+'/'+name+'.png')


def crop_cube_pic(filepath):
    # 将图片切割开
    filenames = os.listdir(filepath)
    for i,file in enumerate(filenames):
        imgpath = filepath + file
        name = file[:-4]
        crop_img(imgpath, name)


def main1():
    #注意：遇到执行命令行的时候需要先注释掉下面的代码，不然未等到运行完，结果可能错误
    #第一步，把全景图全部转换为球面展开图，未切割。手动命令行执行
    print('开始第一步。')
    # python ./py360convert/convert360 --convert e2c --i ./py360convert/test/360/ --o ./py360convert/test/convert_save/ --w 480
    #清除./py360convert/test/convert_save的DS,
    #查看ls -a 清除rm .DS_Store 返回cd ..
    #pip3 install --upgrade opencv-python==3.4.2.16
    #pip3 install --upgrade opencv-contrib-python==3.4.2.16

    # 输入ok，第二步，把球面展开图全部切割，分为黑屏和非黑屏分别保存./test/notblack/ 和./test/black/
    # 清除./py360convert/test/notblack和./py360convert/test/black的DS,
    # 查看ls -a 清除rm .DS_Store 返回cd ..
    print('开始第二步。')
    while(1):
        inpuT=input("命令行运行完毕后需要输入 ok 表示进入步骤2:")
        if inpuT=='ok':
            crop_cube_picpath = './test/convert_save/'
            crop_cube_pic(crop_cube_picpath)
            break

    #第三步，将切割后的全部图片分别边缘处理，采用边界均值填充，四周增大100像素
    print('开始第三步。')
    while(1):
        inpuT=input("命令行运行完毕后需要输入 ok 表示进入步骤3:")
        if inpuT=='ok':
            noblack = './test/notblack/'
            black = './test/black/'
            add_copy_bound_cube(noblack, noblack)
            break

    #第四步，对noblack的文件夹批量风格化，手动执行命令行。转后后文件在 ./test/cube_trans
    print('开始第四步。')
    #python evaluate.py --checkpoint ./checkpoint/wave.ckpt --in-path ./py360convert/test/notblack/ --out-path ./py360convert/test/cube_trans --device /gpu:0
    # 清除./py360convert/test/cube_trans和./py360convert/test/cube_transDel的DS,
    # 查看ls -a 清除rm .DS_Store 返回cd ..

    #第五步，将风格化后的图片切割，切掉边缘的100像素，切割后将黑屏与切割后的图片汇总在 ./test/cube_transDel/
    print('开始第五步。')
    while (1):
        inpuT = input("命令行运行完毕后需要输入 ok 表示进入步骤5:")
        if inpuT == 'ok':
            cube_trans_del_path = './test/cube_transDel/'
            noblackpath = './test/cube_trans/'
            del_bound_cube(noblackpath, black, cube_trans_del_path)
            break
    #第六步，将全部图片重新拼接成球面展开图，最终球面展开图在 ./test/cube_joint/
    print('开始第六步。')
    cube_joint_path='./test/cube_joint/'
    joint_cube_img_allin( cube_trans_del_path,cube_joint_path)
    #第七步，将球面图转换为全景图，手动执行命令行 结果在  ./test/cube_output/
    print('开始第七步。')
    # 清除./py360convert/test/cube_joint的DS,
    # 查看ls -a 清除rm .DS_Store 返回cd ..
    #python ./py360convert/convert360 --convert c2e --i ./py360convert/test/cube_joint/ --o ./py360convert/test/cube_output/ --w 1920 --h 960

def main2():
    #注意：遇到执行命令行的时候需要先注释掉下面的代码，不然未等到运行完，结果可能错误
    #第一步，把全景图错位切开
    print('开始第一步。')
    pic1 = './test/360/'
    pic2 = './test/sift_crop/'
    combinimg.crop_img_SIFT(pic1, pic2)
    #combinimg.crop_img_SIFT_360(pic1, pic2)
    #exit(1)
    # 第二步，风格化.
    print('开始第二步。')
    #python evaluate.py --checkpoint ./checkpoint/wave.ckpt --in-path ./py360convert/test/sift_crop/ --out-path ./py360convert/test/sift_styled --device /gpu:0

    # 第三步，图片和成
    print('开始第三步。')
    while(1):
        inpuT=input("命令行运行完毕后需要输入 ok 表示进入步骤3:")
        if inpuT=='ok':
            inputpath = './test/sift_styled/'
            output = './test/sift_output/'
            names = os.listdir(inputpath)
            combinimg.combin_All(inputpath, names, output)
            break
    print('完毕。输出在./test/sift_output/')

def main3():
    #注意：遇到执行命令行的时候需要先注释掉下面的代码，不然未等到运行完，结果可能错误
    #第一步，把全景图错位切开
    #清除./py360convert/test/360和./py360convert/test/sift_crop的DS
    #查看ls -a 清除rm .DS_Store 返回cd ..
    print('开始第一步。')
    pic1 = './test/360/'
    pic2 = './test/sift_crop/'
    #combinimg.crop_img_SIFT(pic1, pic2)
    combinimg.crop_img_SIFT_360(pic1, pic2)
    #exit(1)
    # 第二步，风格化.
    print('开始第二步。')
    #python evaluate.py --checkpoint ./checkpoint/wave.ckpt --in-path ./py360convert/test/sift_crop/ --out-path ./py360convert/test/sift_styled --device /gpu:0
    #清除./py360convert/test/sift_styled的DS
    #查看ls -a 清除rm .DS_Store 返回cd ..

    # 第三步，图片和成
    print('开始第三步。')
    while(1):
        inpuT=input("命令行运行完毕后需要输入 ok 表示进入步骤3:")
        if inpuT=='ok':
            inputpath = './test/sift_styled/'
            output = './test/sift_output/'
            names = os.listdir(inputpath)
            combinimg.combin_All_new360(inputpath, names, output)
            break
    print('完毕。输出在./test/sift_output/')

def step_one():
    path = './test/360/'
    path_save = './test/simple_concat/'
    for i,file in enumerate(os.listdir(path)):
        img = Image.open(path + file)
        size = img.size
        target = Image.new('RGB',(int(size[0]*1.2),int(size[1])))

        img1 = img.crop((0,0,size[0]*0.1,size[1]))
        img2 = img.crop((size[0]*0.9,0,size[0],size[1]))

        target.paste(img2,(0,0,int(size[0]*0.1),int(size[1])))
        target.paste(img,(int(size[0]*0.1),0,int(size[0]*1.1),int(size[1])))
        target.paste(img1,(int(size[0]*1.1),0,int(size[0]*1.2),int(size[1])))

        target.save(path_save+file)


# 将target文件夹的图片进行全部风格迁移，最后放入target_styled文件夹


# 注释掉stepone，运行这个stepthree，切割，最终图片输出到target_output_final文件夹
def step_three():
    path = './test/concat_styled/'
    path_save = './test/simple_output/'
    for i,file in enumerate(os.listdir(path)):
        target_img = Image.open(path + file)
        size = (1920,960)
        target_crop = target_img.crop((int(size[0]*0.1),0,int(size[0]*1.1),int(size[1])))
        target_crop.save(path_save + file)

def main4():
    print('开始第一步。')
    step_one()
    print('开始第二步。')
    # 风格迁移
    #python evaluate.py --checkpoint ./checkpoint/wave.ckpt --in-path ./py360convert/test/simple_concat/ --out-path ./py360convert/test/concat_styled --device /gpu:0
    while (1):
        inpuT = input("风格化行运行完毕后需要输入 ok 表示进入步骤3:")
        if inpuT == 'ok':
            print('开始第三步。')
            step_three()
            break


def main5():
    print('直接风格迁移。')
    # 风格迁移
    #python evaluate.py --checkpoint ./checkpoint/wave.ckpt --in-path ./py360convert/test/360/ --out-path ./py360convert/test/direct_output --device /gpu:0
#wreck
#udnie
#scream
#la_muse

if __name__ == '__main__':
    #清除./py360convert/test/360的DS
    #查看ls -a 清除rm .DS_Store 返回cd ..
    while True:
        print('#######################')
        print('0、Direct')
        print('1、Simple')
        print('2、SIFT')
        #print('2、Gaussian filter')
        print('3、Filling mean value')
        a= input('请选择需要的功能：')
        if a=='0':
            main5()
        if a=='2':
            main3()
        if a=='1':
            main4()
        # elif a=='2':
        #     img = cv2.imread("./result/360_combin/11.png", cv2.IMREAD_COLOR)
        #     ft.contrast_img(img, 1.3, 3)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        elif a=='3':
            main1()
        else :
            print('输入错误！')




