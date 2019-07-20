# **tvm_106landmarks**
## **1. models comes from zqcnn by zuoqing:**
## https://github.com/zuoqing1988/ZQCNN
## **2. all models in models folder, test images in images folder**
## **3. test different models:**
### **(1) you need to change landmark.py in lines 78~80:**
### >> landmark_graph = './models/48_normal.json'
### >> landmark_lib = './models/48_normal.so'
### >> landmark_param = './models/48_normal.params'
### **TO the models you want to test**
### **(2) change the input size in line 41:**
### >> data_shape = (1,3,48,48)
### **for example, if you want to test model 96.json, 96.so, 96.params**
### **you need change size 48, 48 to 96 x 96**
## 4. test result:
## ![图片](https://github.com/MirrorYuChen/tvm_106landmarks/blob/master/images/result.jpg)
