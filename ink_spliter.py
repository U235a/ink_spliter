# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 07:49:14 2021

@author: U235
"""
import argparse
import glob
import cv2
import numpy as np
import os.path
import sys


def process(path, color_path, bw_path, black_in_area, paper_area, saturate_level):
    # пути к сканам и результатам (пути не должны содержать кирилицу, из-за OpenCV)
    #path = "ink_scans\\"
    #color_path="color_layer\\"
    #bw_path="bw_layer\\"

    # ПАРАМЕТРЫ:
    #black_ink_area=1 # минимальный процент черной краски на скане
    #papper_area=85 # минимальный процент чистой бумаги на скане
    #saturate_level=98 # уровень насыщенности краски на скане
    #scale = 0.25 # коэффициент масштабиравания для скорости обработки

    files = glob.glob(path+'*.tif')
    black_inks = []
    pappers = []
    color_inks = []

    # первый проход: анализ сканов, поиск плашечного цвета
    for img_file in files:
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        margin=int(0.05*width) # неучитываемые поля 5%
        resized = cv2.resize(img[margin:-margin, margin:-margin], (width, height), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        saturate = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)[:, :, 1] # насыщеность цвета
        bw_levels = np.percentile(gray, [black_ink_area, papper_area])
        saturate[gray < bw_levels[0]] = 0
        saturate[gray > bw_levels[1]] = 0
        max_saturate = np.percentile(saturate, saturate_level)
        black_ink = np.median(resized[gray < bw_levels[0]], axis=0)
        papper = np.median(resized[gray > bw_levels[1]], axis=0)
        color_ink = np.median(resized[saturate > max_saturate], axis=0)
        black_inks.append(black_ink)
        pappers.append(papper)
        color_inks.append(color_ink)

    # расчет усредненных цветов, как медианы по сканам:
    c = np.single(np.median(color_inks, axis=0))
    w = np.single(np.median(pappers, axis=0))
    b = np.single(np.median(black_inks, axis=0))
    print(f'color ink:\n{c[::-1]}\npapper:\n{w[::-1]}\nblack_ink:\n{b[::-1]}')

    # Расчет матрицы преобразования цвета
    v1=np.cross(np.cross(w-b, c-b), w-b)
    v1=v1/np.linalg.norm(v1)
    v1=255*v1/np.dot(c-b, v1)

    v2=np.cross(np.cross(c-w, b-w), c-w)
    v2=v2/np.linalg.norm(v2)
    v2=255*v2/np.dot(b-w, v2)

    v3=np.array([[-v1[0], -v1[1], -v1[2], 255+np.dot(b, v1)]])
    v4=np.array([[-v2[0], -v2[1], -v2[2], 255+np.dot(w, v2)]])
    M=np.vstack((v3, v4))
    print(f'Color matrix transformation (BGR order and bias):\n{M}')

    # второй проход: обработка, создание двух изображений - слоев
    print('Processing:')
    for img_file in files:
        name=os.path.split(img_file)
        print(name[1])
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        new_img=cv2.transform(img, M)
        cv2.imwrite(os.path.join(color_path, name[1].replace('.tif','_layer_c.png')), new_img[:,:,0])
        cv2.imwrite(os.path.join(bw_path, name[1].replace('.tif','_layer_b.png')), new_img[:,:,1])
    print('Done')
    

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="пути к сканам", default='ink_scans')
    parser.add_argument('--color_path', help="пути к результатам", default='color_layer')
    parser.add_argument('--bw_path', default='bw_layer')
    parser.add_argument('--black_ink_area', default=1, type=int)
    parser.add_argument('--paper_area', default=85, type=int)
    parser.add_argument('--saturate_level', default=98, type=int)
    parser.add_argument('--scale', default=0.25, type=float)
    process(**vars(parser.parse_args()))
 

    
'__main__' == __name__ and main(sys.argv[1:])
