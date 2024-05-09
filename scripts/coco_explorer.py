import os
from PIL import Image
import cv2
import numpy as np
from pycocotools.coco import COCO
import logging
import shutil
import json
from pathlib import *

palette=[
            [128, 0, 0],
            [128, 64, 128],
            [192, 0, 192],
            [0, 128, 0],
            [128, 128, 0],
            [64, 64, 0],
            [64, 0, 128],
            [0, 0, 0]
        ]

not_in_p = []

def explore_file(path:str) -> None:
    
    img = Image.open(path)
    img_arr = np.asarray(img, dtype=np.uint8).tolist()
    #print(np.asarray(img, dtype=np.uint8).shape)
    converted_list = []
    # print(img_arr)
    # print(len(img_arr))
    for horizontal in img_arr:
        for rgb in horizontal:
            #print(palette.index(rgb))
            converted_list.append(palette.index(rgb))
    converted_arr = np.asarray(converted_list, dtype=np.uint8).reshape([2160,4096])
    converted_img = Image.fromarray(converted_arr).convert('P')
    converted_img.putpalette(np.array(palette, dtype=np.uint8))
    p = "../UAVID/uavid_converted/annotations/test2017/3.png"
    converted_img.save(p)

    img_arr_test = np.asarray(Image.open(p), dtype=np.uint8)
    print(img_arr_test)
    #lst = img_arr_test.tolist()
    #print(lst)
    #print(np.asarray(img_arr, dtype=np.uint8))
    # arr = Image.fromarray(img_arr).convert('P')
    # arr2 = np.asarray(arr, dtype=np.uint8)
    # print(arr2)
    # for arr in arr2:
    #     for arr3 in arr:
    #         if arr3 not in not_in_p:
    #             not_in_p.append(arr3)
    # print(not_in_p)

def numpy_test():
    a = np.arange(1,16)
    print(a)
    b = a.reshape(3,5)
    print(b)
    c = b.reshape(3,5,3)
    print(c)


def convert_test():
    arr=np.array([[128, 0, 0],
            [128, 64, 128],
            [192, 0, 192],
            [0, 128, 0],
            [128, 128, 0],
            [64, 64, 0],
            [64, 0, 128],
            [0, 0, 0]], dtype=np.uint8)
    print(arr)
    img = Image.fromarray(arr).convert('P')
    img_arr = np.asarray(img, dtype=np.uint8)
    print(img_arr)


#numpy_test()

path = "../UAVID/uavid_converted/annotations/test2017/280_000000_src.png"
explore_file(path)
#convert_test()

# seg_map = np.zeros((2160,4096),dtype=np.uint8)

# for subfolder,dirs,files in os.walk(path):
#     for file in files:
#         filename = file.split("src")[0]+"_src.png"
#         shutil.move(subfolder+'/'+file, subfolder+'/'+filename)