import os
from PIL import Image
import numpy as np
import time

PALETTE=[[128, 0, 0],
            [128, 64, 128],
            [192, 0, 192],
            [0, 128, 0],
            [128, 128, 0],
            [64, 64, 0],
            [64, 0, 128],
            [0, 0, 0]]

PT_DICT = {}

def build_dictionary():
    count = 0
    for i in PALETTE:
        PT_DICT[str(i)] = count
        count += 1
    


def convert_file(subdir: str, file: str):
    
    print("Current file: "+ file + " (" + subdir.split('/')[-1]+")")
    start = time.time()
    image_lst = np.asarray(Image.open(subdir+'/'+file),dtype=np.uint8)
    #shape = image_lst.shape
    #prod = shape[0]*shape[1]
    
    #image_lst = image_lst.flatten().reshape((prod, 3))
    # converted = [PT_DICT[str(item.tolist())] for item in image_lst]

    image_lst = image_lst.tolist()
    converted = []
    for horizontal in image_lst:
        for rgb in horizontal:
            # print(palette.index(rgb))
            converted.append(PALETTE.index(rgb))
    length = len(converted)
    print(time.time()-start)
    if(length == 3840*2160):
        converted_arr = np.asarray(converted, dtype=np.uint8).reshape([2160,3840])
    elif(length == 4096*2160):
        converted_arr = np.asarray(converted, dtype=np.uint8).reshape([2160,4096])
    else:
        print("Skipping: "+ file + " (" + subdir.split('/')[-1]+")")
        return
    print("Converting: "+ file + " (" + subdir.split('/')[-1]+")")
    converted_img = Image.fromarray(converted_arr).convert('P')
    converted_img.putpalette(np.array(PALETTE, dtype=np.uint8))

    components = file.split('_')
    filepath = subdir+'/'+components[0]+'_'+components[1]+'.png'
    converted_img.save(filepath)

    ###### MMSEG

    mask = np.array(Image.open(filepath))
    mask_cp = mask.copy()
    filepath = filepath.split('.')[0]+'_labelTrainIds.png'
    Image.fromarray(mask_cp).save(filepath, 'PNG')
    

def convert_all():
    build_dictionary()
    path = "/home/nrodenbuesch/Studienarbeit/UAVID/uavid_converted/annotations/test2017"
    for subdir, dirs, files in os.walk(path):
        if files != []:
            files = sorted(files)
            for file in files:
                if file.split('_')[-1] == "src.png":
                    convert_file(subdir, file)


convert_all()