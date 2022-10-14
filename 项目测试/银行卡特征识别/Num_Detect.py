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

def tem_pretreat(img):
    gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, threshold= cv2.threshold(img,20,255,cv2.THRESH_BINARY_INV)
    return threshold

if __name__ == '__main__':
    args = Set_argparse()
    temple = cv2.imread(args["template"])
    show_cv(temple,"template")

    th_tem = tem_pretreat(temple)
    show_cv(th_tem,"th_template")

