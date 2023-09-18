# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:47:52 2023

@author: SATYAKI
"""

import numpy as np
from PIL import Image

np.random.seed(9)

mat1 = np.random.randint(20,size=(100,100))

def shanon_entropy(arr:np.array) -> float:
    _,counts = np.unique(arr,return_counts=True)
    probabilities = counts / len(arr)
    return -np.sum(probabilities * np.log2(probabilities))

def entropy_array(mat:np.array,base:int) -> np.array:
    i,j = mat.shape
    list1 = list()
    x,y,prev_x,prev_y = base,base,0,0
    while x<=i:
        while y<=j:
            se=shanon_entropy(mat[prev_x:x,prev_y:y].flatten())
            list1.append(se)
            y += base
            prev_y += base
        prev_x +=base
        x += base
        y=base
        prev_y=0
    return np.array(list1)

def entropy_image() -> np.array:
    img = Image.open("./images/0.pgm")
    img_arr = np.asarray(img)
    en = entropy_array(img_arr, 5)
    return en

def save_to_file(en_arr:np.array):
    np.savetxt('entopy.txt',en_arr,newline=" ")

    
    
if __name__ == "__main__":
    # res = shanon_entropy(arr)
    # print(f"entopy : {res}")
    # en_arr = entropy_array(mat1,7)
    # print(mat1)
    # print(en_arr, len(en_arr))
    en_arr = entropy_image()
    print(en_arr , en_arr.shape )
    save_to_file(en_arr=en_arr)
    
    