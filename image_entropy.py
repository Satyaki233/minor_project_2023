import numpy as np
import os
from PIL import Image


class Entropy:
    def __init__(self,path:str):
        self.path = path
    def shanon_entropy(self,arr: np.array) -> float:
        _, counts = np.unique(arr, return_counts=True)
        probabilities = counts / len(arr)
        return -np.sum(probabilities * np.log2(probabilities))

    def entropy_array(self,mat: np.array, base: int) -> np.array:
        i, j = mat.shape
        list1 = list()
        x, y, prev_x, prev_y = base, base, 0, 0
        while x <= i:
            while y <= j:
                se = self.shanon_entropy(mat[prev_x:x, prev_y:y].flatten())
                list1.append(se)
                y += base
                prev_y += base
            prev_x += base
            x += base
            y = base
            prev_y = 0
        return np.array(list1)

    def entropy_image(self,file_name: str) -> np.array:
        img = Image.open(file_name)
        img_arr = np.asarray(img)
        en = self.entropy_array(img_arr, 5)
        return en

    def save_to_file(self,en_arr: np.array, file_name:str):
        entopy_file="entropy.txt"
        file = open(entopy_file, 'a')

        file.write(file_name)
        file.write("\n")
        for i in en_arr:
            file.write(str(i) + ",")
        file.write("\n")

        file.close()

    def read_to_file(self) -> tuple[np.array,np.array]:
        key = []
        value = []
        file = open('entropy.txt', 'r')
        count = 0
        k = ""
        v = ""
        while True:
            k = str(file.readline())
            v = str(file.readline())
            temp=[]
            if not k:
                break
            else:
                key.append(k[:len(k) - 1])
                for i in list(v[:len(v) - 2].split(",")):
                    temp.append(float(i))
                value.append(temp)

        file.close()
        return (np.array(key), np.array(value))

    def image_entropy(self,path:str,folder:str=None):
        list_of_file=os.listdir(path+folder)
        for name in list_of_file:
            en_arr = self.entropy_image(path+folder+name)
            self.save_to_file(en_arr=en_arr,file_name=folder)





def calculate_entropy():
    if os.path.exists("entropy.txt"):
        os.remove("entropy.txt")
    path = "./images/"
    list_of_file_name = os.listdir(path)

    obj=Entropy(path)
    if os.path.exists("entropy.txt"):
        os.remove("entropy.txt")
        
    for name in list_of_file_name:
        en_arr = obj.entropy_image(path+name)
        # print(en_arr, en_arr.shape)
        obj.save_to_file(en_arr=en_arr, file_name=name)
    key, value = obj.read_to_file()
    print(type(key))
    for i in range(len(key)):
        print(key[i])
        print(len(value[i]))

def calculate_entropy_02():
    if os.path.exists("entropy.txt"):
        os.remove("entropy.txt")
    path="./database/face_database/"
    obj = Entropy(path=path)
    x = os.listdir(path)
    for i in x:
        
        if "s" in i:
            y=os.listdir(path+i)
            for j in y:
                en_arr = obj.entropy_image(path+i+'/'+j)
                obj.save_to_file(en_arr=en_arr,file_name=i)

def get_entropy():
    en = Entropy("./face_database/")
    return en.read_to_file()


if __name__ == "__main__":
    calculate_entropy_02()
    