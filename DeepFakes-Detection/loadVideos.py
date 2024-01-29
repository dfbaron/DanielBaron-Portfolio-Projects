# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:51:20 2021

@author: DanielBaron
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json
import pandas as pd
import time

t1 = time.time()

os.chdir(r'C:\Users\DanielBaron\Desktop\CursosMaestria\Semestre I\Machine Learning Techniques\Proyecto')

metadata_file_name = 'metadata.json'
with open(os.path.join('dataset', metadata_file_name)) as file:
    metadata = json.loads(file.read())

data = []
for key in metadata:
    data.append([key, metadata[key]['label'], metadata[key]['split']])

def load_video(path):
    cap = cv2.VideoCapture(path)
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

path = os.path.join('dataset', data[10][0])
numpy_array = load_video(path)

print(time.time() - t1)