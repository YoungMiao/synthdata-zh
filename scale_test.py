#coding: utf-8
import os
import Image
import sys 
import cv2
import chardet 
import numpy as np
reload(sys)  
sys.setdefaultencoding('utf8')
#if not os.path.exists('./R_test'):
#    os.makedirs('./R_test')
BGS_DIR = './bgs_200'
fname = BGS_DIR 
filenames = os.listdir(BGS_DIR)
for fn in filenames:
    fullfilename = os.path.join(BGS_DIR,fn)
    print(fullfilename)
    bg = cv2.imread(fullfilename, cv2.CV_LOAD_IMAGE_COLOR)
    imgH = 500
    h, w = bg.shape[:2]
    ratio = w / float(h)
    imgW = int(ratio * imgH)
    res=cv2.resize(bg,(imgW,imgH),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(fullfilename, res)
