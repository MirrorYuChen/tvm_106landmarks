import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime

import cv2 as cv
import os

class FaceLandmarks:
    '''
    人脸关键点定位模块
    '''
    def __init__(self, ctx,landmark_graph,landmark_lib,landmark_param):
        self.ctx = ctx
        self.landmark_graph =landmark_graph
        self.landmark_lib =landmark_lib
        self.landmark_param = landmark_param
        self.initialized = False
        self.load_model()

    '''
    模型载入：使用人脸关键点定位功能时，需要载入关键点定位模型：json文件、lib文件和param文件，及对应运行平台
    '''
    def load_model(self):
        if os.path.exists(self.landmark_graph) and os.path.exists(self.landmark_lib) and os.path.exists(self.landmark_param):
            print(self.landmark_graph)
            print(self.landmark_lib)
            print(self.landmark_param)
            loaded_graph = open(self.landmark_graph).read()
            loaded_lib = tvm.module.load(self.landmark_lib)
            loaded_param = bytearray(open(self.landmark_param, "rb").read())
            self.module = graph_runtime.create(loaded_graph, loaded_lib, self.ctx)
            self.module.load_params(loaded_param)
            self.initialized = True

    '''
    预处理：对当前传入人脸进行去均值和归一化操作
    '''
    def prepocess(self, face):
        data_shape = (1,3,48,48)
        resized_face = cv.resize(face, (data_shape[2], data_shape[3]))
        resized_face = np.float32(resized_face)
        resized_face = np.transpose(np.array(resized_face), (2, 0, 1))
        resized_face = (resized_face - 127.5) * 0.0078125
        return resized_face

    '''
    关键点检测：传入待检测人脸face，传出关键点对应数组[(x0, y0), (x1, y1), ..., (x105, y105)]
    '''
    def detect(self, face):
        if not self.initialized:
            print("landmark model uninitialized.")
            return None
        if face is not None:
            resized_face = self.prepocess(face)
            face_input = tvm.nd.array(resized_face.astype("float32"))
            self.module.run(data=face_input)
            landmarks = self.module.get_output(0).asnumpy()[0]
            keypoints = self.calc_keypoints(landmarks,face)
            return keypoints
        else:
            print('Empty face')
            return None
    def calc_keypoints(self,landmarks,face):
        keypoints =[]
        num_keypoints = len(landmarks) // 2
        print('num keypoints: {}'.format(num_keypoints))
        for j in range(num_keypoints):
            x = abs(landmarks[2 * j] * face.shape[1])
            y = abs(landmarks[2 * j + 1] * face.shape[0])
            keypoints.append((x, y))
        return keypoints
        
if __name__ == '__main__':
    face = cv.imread('./images/test.jpg')
    ctx = tvm.cpu(0)
    landmark_graph = './models/48_normal.json'
    landmark_lib = './models/48_normal.so'
    landmark_param = './models/48_normal.params'
    face_landmark =  FaceLandmarks(ctx, landmark_graph, landmark_lib, landmark_param)
    keypoints = face_landmark.detect(face)
    for pt in keypoints:
        cv.circle(face, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), 2)
    cv.imwrite('./images/result.jpg', face)
         
        
    
    
