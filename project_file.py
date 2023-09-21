import cv2, time
import mediapipe as mp
import numpy as np
from PIL import Image
import os


class FaceDetector:
    def __init__(self, minDetectionCon=0.75):

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDrow = mp.solutions.drawing_utils
        self.faceDectection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFace(self, img, drow=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDectection.process(imgRGB)
        # print(self.result)

        bboxs = []
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                cv2.rectangle(img, bbox, (255, 0, 255), 2)

                img = self.fancyDrow(img, bbox)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 0), 2)

        return img, bboxs

    def fancyDrow(selfself, img, bbox, l=30, t=3):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 255, 255), 1)

        # top left
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # top right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


class Entropy:
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
        img = Image.open("dataset/" + str(count) + ".jpg")
        img_arr = np.asarray(img)
        en = self.entropy_array(img_arr, 5)
        return en

    def save_to_file(self,en_arr: np.array, count: int):
        file = open('entopy.txt', 'a')

        file.write("IMG-" + str(count))
        file.write("\n")
        for i in en_arr:
            file.write(str(i) + ",")
        file.write("\n")

        file.close()

    def read_to_file(self) -> tuple:
        key = []
        value = []
        file = open('entopy.txt', 'r')
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


def main_1():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    count = 0
    frame_rate = 2
    prev = 0
    while True:
        time_elapsed = time.time() - prev
        success, img = cap.read()
        img, bbox = detector.findFace(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(bbox)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Controlling the frame rate
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            if bbox:
                for box in bbox:
                    x, y, w, h = box[1]
                    if x < 0 and y < 0:
                        face_image = gray[0:0 + h, 0:0 + w]
                        cv2.imwrite('./dataset/' + str(count) + '.jpg', face_image)
                        count += 1
                    elif x < 0:
                        face_image = gray[y:y + h, 0:0 + w]
                        cv2.imwrite('./dataset/' + str(count) + '.jpg', face_image)
                        count += 1
                    elif y < 0:
                        face_image = gray[0:0 + h, x:x + w]
                        cv2.imwrite('./dataset/' + str(count) + '.jpg', face_image)
                        count += 1
                    else:
                        face_image = gray[y:y + h, x:x + w]
                        cv2.imwrite('./dataset/' + str(count) + '.jpg', face_image)
                        count += 1

        cv2.imshow("image", img)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break


def main_2():
    folder = os.listdir("./dataset")
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


if __name__ == '__main__':
    main_2()