import cv2 as cv
import os
import glob
import numpy as np

video_name = '/mnt/lustre/hnqiu/data/actor_align_512/Actor_*/*.mp4'

files = sorted(glob.glob(video_name))

for index in range(len(files)):
    file = files[index]
    file1 = file.replace('.mp4', '/').replace('actor_align_512', 'actor_align_512_png')
    if not os.path.exists(file1):
        os.makedirs(file1)
    cap=cv.VideoCapture(file)
    isOpened=cap.isOpened()
    i=0
    while(isOpened) and i < 9000:
        i=i+1
        flag,frame=cap.read()
        # fileName = '%03d'%i+".jpg"
        name = ('00000' + str(i))[-5:]
        if flag == True :
            cv.imwrite(file1 + name + ".png", frame)
            cv.waitKey(1)
        else:
            break
    cap.release()
print('end')