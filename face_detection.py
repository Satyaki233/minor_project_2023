import cv2,time
import mediapipe as mp
import os,progressbar


class FaceDetector:
    def __init__(self, minDetectionCon=0.75):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDrow = mp.solutions.drawing_utils
        self.faceDectection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        # self.detectionScore = 0

    def findFace(self, img, drow=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDectection.process(imgRGB)
        # print(self.result)

        bboxs = []
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                detection_scrore = int(detection.score[0] * 100)
                # if detection_scrore < 70:
                #     continue
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                cv2.rectangle(img, bbox, (255, 0, 255), 2)

                img = self.fancyDrow(img, bbox)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 0), 2)
                # self.detectionScore = int(detection.score[0]*100)

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

def detect_face(path:str=None,video_name:str=None):
    if(path != None):
        cap=cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    
    count = 0
    frame_rate = 2
    prev = 0
    while True:
        time_elapsed = time.time() - prev
        success, img = cap.read()
        if success == False: break
        img, bbox = detector.findFace(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # #print(detector.detectionScore)
        # if detector.detectionScore < 70:
        #     continue
        # print(bbox)

        # cTime = time.time()
        # if cTime - pTime == 0.0:
        #     continue
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Controlling the frame rate
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            if bbox:
                for box in bbox:
                    x, y, w, h = box[1]
                    img_name ="s"+ video_name +'_'+ str(count)
                    if x < 0 and y < 0:
                        face_image = gray[0:0 + h, 0:0 + w]
                        face_image=cv2.resize(face_image,(128,128))
                        cv2.imwrite('./images/' + img_name + '.pgm', face_image,
                        [cv2.IMWRITE_PXM_BINARY,0])
                        count += 1
                    elif x < 0:
                        face_image = gray[y:y + h, 0:0 + w]
                        face_image = cv2.resize(face_image, (128, 128))
                        cv2.imwrite('./images/' + img_name + '.pgm', face_image,
                        [cv2.IMWRITE_PXM_BINARY,0])
                        count += 1
                    elif y < 0:
                        face_image = gray[0:0 + h, x:x + w]
                        face_image = cv2.resize(face_image, (128, 128))
                        cv2.imwrite('./images/' + img_name + '.pgm', face_image,
                        [cv2.IMWRITE_PXM_BINARY,0])
                        count += 1
                    else:
                        face_image = gray[y:y + h, x:x + w]
                        face_image = cv2.resize(face_image, (128, 128))
                        cv2.imwrite('./images/' + img_name + '.pgm', face_image,
                        [cv2.IMWRITE_PXM_BINARY,0])
                        count += 1
        
        cv2.imshow("video : " + video_name, img)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_face_from_video(path,load_file:bool=True):
    save_file_name = 'savefile.txt'
    x = os.listdir(path)
    progress,video_number = 0,0
    unextracted_video =list()
    widgets=[progressbar.Percentage(), " ", progressbar.GranularBar()," ",progressbar.ETA() ]
    bar = progressbar.ProgressBar(max_value=len(x),widgets=widgets).start()
    try:
        print("\nlength of file :" + str(len(x)))
        if os.path.exists(save_file_name) and load_file is True:
            with open(save_file_name) as file:
                unextracted_video = [line.strip() for line in file]
            progress = len(x) - len(unextracted_video)
        else:
            unextracted_video = [o for o in x if '.' not in o ]
        print("\nunextracted videos : \n", unextracted_video)
        for i in unextracted_video:           
            y = os.listdir(path+'/'+i)
            for j in y:
                if ".avi" in j:
                    video_number=i
                    detect_face(path+'/'+i+'/'+j,i)
                    bar.update(progress)
                    progress+=1
                    unextracted_video.remove(video_number)
                        
    except KeyboardInterrupt:
        
        if os.path.exists(save_file_name):
            os.remove(save_file_name)
        print(f"\nlast video number : {video_number}")
        print("remaing files : \n" ,unextracted_video)
        file = open(save_file_name,"a")
        for i in unextracted_video:
                file.write(str(i) + "\n")
        file.close()





if __name__=='__main__':
    # detect_face("./database/video_database/vid.avi")
    detect_face_from_video("./database/video_database")