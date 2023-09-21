import numpy as np
import os
from PIL import Image


class Entropy:
    def __init__(self,filename:str="entropy_dataset.txt"):
        self.filename = filename

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

    def entropy_image(self,count: int) -> np.array:
        img = Image.open("images/" + str(count) + ".pgm")
        img_arr = np.asarray(img)
        en = self.entropy_array(img_arr, 5)
        return en

    def save_to_file(self,en_arr: np.array, count: int):
        file = open(self.filename , 'a')
        
        file.write("IMG-" + str(count))
        file.write("\n")
        for i in en_arr:
            file.write(str(i) + ",")
        file.write("\n")

        file.close()

    def read_to_file(self) -> tuple:
        key = []
        value = []
        file = open(self.filename, 'r')
        count = 0
        k = ""
        v = ""
        while True:
            k = str(file.readline())
            v = str(file.readline())

            if not k:
                break
            else:
                key.append(k[:len(k) - 1])
                value.append(list(v[:len(v) - 2].split(",")))
                

        file.close()
        return (key, value)


def main_2():
    folder = os.listdir("./images")
    count = len(folder)
    i = 0
    obj=Entropy()
    while i < count:
        en_arr = obj.entropy_image(i)
        # print(en_arr, en_arr.shape)
        obj.save_to_file(en_arr=en_arr, count=i)
        i += 1
    key, value = obj.read_to_file()
    print(key)
    for i in range(len(key)):
        print(key[i])
        print(value[i])

if __name__ == "__main__":
    main_2()