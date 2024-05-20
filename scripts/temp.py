import numpy as np
from PIL import Image
import os


#Nötige Imports
palette=[[255,0,0],...] #Liste mit den ursprünglichen RGB-Farben

def convert_file(file_path: str):
    img = Image.open(path)
    img_arr = np.asarray(img, dtype=np.uint8).tolist()
    converted_list = []
    for horizontal in img_arr:
        for rgb in horizontal:
            converted_list.append(palette.index(rgb))
    converted_arr = np.asarray(converted_list, dtype=np.uint8).reshape(
        np.asarray(img, dtype=np.uint8).shape)
    converted_img = Image.fromarray(converted_arr).convert('P')
    new_path = '..' + path.split('.')[2] + '_labelTrainIds.png' #Segmentationmask Suffix
    converted_img.save(new_path)

path = "../Studienarbeit/UAVID/annotations/"
for subfolder, directories, files in os.walk(path):
    for file in files:
        convert_file(subfolder + '/' + file)





