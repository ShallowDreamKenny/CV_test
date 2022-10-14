#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Num_Detect.py
# Time       ：10/14/22 5:26 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def Set_argparse():
    """
    初始化参数
    :return: 参数的值
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default="./pic/card1.png",help="path of the input image")
    ap.add_argument("-t", "--template", default="./pic/model.png",help="path of the input model")
    args = vars(ap.parse_args())
    return args

def show_cv(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = Set_argparse()
    img = cv2.imread(args["template"])
    show_cv(img,"template")

