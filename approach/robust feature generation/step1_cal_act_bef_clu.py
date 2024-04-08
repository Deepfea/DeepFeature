import os
import numpy as np
import skimage.io as io
from keras.models import load_model
from keras import Model

def load_mnsit_act(path, savePath, model):
    classLsit = os.listdir(path)
    for class1 in classLsit:
        classPath = path + os.sep + class1
        fileList = os.listdir(classPath)
        key = 0
        tempList = []
        for file in fileList:
            filePath = classPath + os.sep + file
            print(filePath)
            img = io.imread(filePath)
            # print(img)
            img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
            img = np.expand_dims(img, axis=0)
            img = img.transpose((1, 2, 0))
            if key == 0:
                tempArr = np.empty((len(fileList), img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
            tempList.append(file)
            tempArr[key] = img
            key = key + 1
        tempArr = tempArr.astype('float32') / 255
        i = Model(inputs=model.layers[0].output, outputs=model.layers[6].output)   #mnist lenet1 6 7 lenet5 8 9
        temp = i.predict(tempArr)
        print(class1 + ":finish!")
        actPath = savePath + os.sep + str(class1) + '_act.npy'
        labelPath = savePath + os.sep + str(class1) + '_label.npy'
        file_name = savePath + os.sep
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        np.save(actPath, temp)
        np.save(labelPath, np.array(tempList))
def load_CIFAR_act(path, savePath, model):
    classLsit = os.listdir(path)
    for class1 in classLsit:
        classPath = path + os.sep + class1
        fileList = os.listdir(classPath)
        key = 0
        tempList = []
        for file in fileList:
            filePath = classPath + os.sep + file
            img = io.imread(filePath)
            if key == 0:
                tempArr = np.empty((len(fileList), img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
            tempList.append(file)
            tempArr[key] = img
            key = key + 1
        tempArr = tempArr.astype('float32') / 255
        print(tempArr)
        i = Model(inputs=model.layers[0].output, outputs=model.layers[22].output) #vgg16 22 23 resnet20 71 72
        # print(model.layers[68].name)
        temp = i.predict(tempArr)
        # temp = i.predict(tempArr).reshape(len(tempArr), -1, model.layers[71].output.shape[-1])
        # print(temp.shape)
        # temp = np.mean(temp, axis=1)
        # print(temp.shape)
        print(temp)
        print(class1 + ":finish!")
        actPath = savePath + os.sep + str(class1) + '_act.npy'
        labelPath = savePath + os.sep + str(class1) + '_label.npy'
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        np.save(actPath, temp)
        np.save(labelPath, np.array(tempList))

def load_Imaget_act(path, savePath, model):
    classLsit = os.listdir(path)
    for class1 in classLsit:
        classPath = path + os.sep + class1
        fileList = os.listdir(classPath)
        key = 0
        tempList = []
        for file in fileList:
            filePath = classPath + os.sep + file
            img = io.imread(filePath)
            if key == 0:
                tempArr = np.empty((len(fileList), img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
            tempList.append(file)
            tempArr[key] = img
            key = key + 1
        tempArr = tempArr.astype('float32') / 255
        i = Model(inputs=model.layers[0].output, outputs=model.layers[176].output) #resnet50 176 177
        temp = i.predict(tempArr)
        print(class1 + ":finish!")
        actPath = savePath + os.sep + str(class1) + '_act.npy'
        labelPath = savePath + os.sep + str(class1) + '_label.npy'
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        np.save(actPath, temp)
        np.save(labelPath, np.array(tempList))

if __name__ == '__main__':
    pass
