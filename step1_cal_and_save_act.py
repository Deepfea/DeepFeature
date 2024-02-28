import os
import numpy as np
import skimage.io as io
from keras.models import load_model
from keras import Model
from skimage.transform import resize
# def load_mnist_act(path, savePath, model, model_name):
#     classLsit = os.listdir(path)
#     for class1 in classLsit:
#         classPath = path + os.sep + class1
#         feaList = os.listdir(classPath)
#         tempList = []
#         tempIndex = []
#         flag = 0
#         for fea in feaList:
#             feaPath = classPath + os.sep + fea
#             fileList = os.listdir(feaPath)
#             key = 0
#             for file in fileList:
#                 filePath = feaPath + os.sep + file
#                 img = io.imread(filePath)
#                 img = img[:, :, 0]
#                 img = np.expand_dims(img, axis=0)
#                 img = img.transpose((1, 2, 0))
#                 if key == 0:
#                     temp = np.empty((len(fileList), img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
#                 tempList.append(file)
#                 tempIndex.append(fea)
#                 temp[key] = img
#                 key = key + 1
#             if flag == 0:
#                 tempArr = temp
#                 flag = 1
#             else:
#                 tempArr = np.concatenate((tempArr, temp))
#         tempArr = tempArr.astype('float32') / 255
#         if model_name == 'lenet1':   #1, 3, Conv;
#             i = Model(inputs=model.layers[0].output, outputs=model.layers[3].output)
#             temp = i.predict(tempArr).reshape(len(tempArr), -1, i.output.shape[-1])
#             temp = np.mean(temp, axis=1)
#         if model_name == 'lenet5':   #1, 3, Conv; 6, 7, dense;
#             i = Model(inputs=model.layers[0].output, outputs=model.layers[3].output)
#             temp = i.predict(tempArr).reshape(len(tempArr), -1, i.output.shape[-1])
#             temp = np.mean(temp, axis=1)
#         print(temp.shape)
#         print(class1 + ":finish!")
#         actPath = savePath + os.sep + str(class1) + '_act_after_clu.npy'
#         labelPath = savePath + os.sep + str(class1) + '_label_after_clu.npy'
#         indexPath = savePath + os.sep + str(class1) + '_index_after_clu.npy'
#         if not os.path.exists(savePath):
#             os.mkdir(savePath)
#         np.save(actPath, temp)
#         np.save(labelPath, np.array(tempList))
#         np.save(indexPath, np.array(tempIndex))
#
# def load_cifar_act(path, savePath, model, model_name):
#     classLsit = os.listdir(path)
#     for class1 in classLsit:
#         # print(class1)
#         classPath = path + os.sep + class1
#         feaList = os.listdir(classPath)
#         tempList = []
#         tempIndex = []
#         flag = 0
#         for fea in feaList:
#             # print(fea)
#             feaPath = classPath + os.sep + fea
#             fileList = os.listdir(feaPath)
#             key = 0
#             for file in fileList:
#                 # print(file)
#                 filePath = feaPath + os.sep + file
#                 img = io.imread(filePath)
#                 if key == 0:
#                     temp = np.empty((len(fileList), img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
#                 tempList.append(file)
#                 tempIndex.append(fea)
#                 temp[key] = img
#                 key = key + 1
#             if flag == 0:
#                 tempArr = temp
#                 flag = 1
#             else:
#                 tempArr = np.concatenate((tempArr, temp))
#         tempArr = tempArr.astype('float32') / 255.0
#         if model_name == 'vgg16':   #16, 17, Conv; 20, 21, Dense;
#             i = Model(inputs=model.layers[0].output, outputs=model.layers[17].output)
#             temp = i.predict(tempArr).reshape(len(tempArr), -1, i.output.shape[-1])
#             temp = np.mean(temp, axis=1)
#         if model_name == 'resnet20':   #64, 68, Conv;
#             i = Model(inputs=model.layers[0].output, outputs=model.layers[68].output)
#             temp = i.predict(tempArr).reshape(len(tempArr), -1, i.output.shape[-1])
#             temp = np.mean(temp, axis=1)
#         print(temp.shape)
#         print(class1 + ":finish!")
#         actPath = savePath + os.sep + str(class1) + '_act_after_clu.npy'
#         labelPath = savePath + os.sep + str(class1) + '_label_after_clu.npy'
#         indexPath = savePath + os.sep + str(class1) + '_index_after_clu.npy'
#         if not os.path.exists(savePath):
#             os.mkdir(savePath)
#         np.save(actPath, temp)
#         np.save(labelPath, np.array(tempList))
#         np.save(indexPath, np.array(tempIndex))

def load_imagenet_act(bath_path, model, model_name):
    class_list = ['cat', 'dog']
    load_fea_path = os.path.join(bath_path, 'fea_select', 'fea')
    for class1 in class_list:
        load_class_path = os.path.join(load_fea_path, class1)
        fea_list = os.listdir(load_class_path)
        temp_list = []
        temp_index = []
        flag = 0
        for fea in fea_list:
            print(fea)
            fea_path = os.path.join(load_class_path, fea)
            file_list = os.listdir(fea_path)
            key = 0
            for file in file_list:
                # print(file)
                file_path = os.path.join(fea_path, file)
                # print(file_path)
                img1 = io.imread(file_path)
                img = resize(img1, (224, 224, 3))
                if key == 0:
                    temp = np.empty((len(file_list), img.shape[0], img.shape[1], img.shape[2]), dtype='float32')
                temp_list.append(file)
                temp_index.append(fea)
                temp[key] = img
                key = key + 1
            if flag == 0:
                temp_arr = temp
                flag = 1
            else:
                temp_arr = np.concatenate((temp_arr, temp))
            # print(temp[0])
            print(temp_arr.shape)
            print(len(temp_list))
            print(len(temp_index))
        if model_name == 'vgg_19':  # 33, 35, Conv; 39, 40, Dense;
            i = Model(inputs=model.layers[0].output, outputs=model.layers[35].output)
            temp_1 = i.predict(temp_arr).reshape(len(temp_arr), -1, i.output.shape[-1])
            temp_1 = np.mean(temp_1, axis=1)
        if model_name == 'resnet_50':  # 169, 173, Conv;
            i = Model(inputs=model.layers[0].output, outputs=model.layers[173].output)
            temp_1 = i.predict(temp_arr).reshape(len(temp_arr), -1, i.output.shape[-1])
            temp_1 = np.mean(temp_1, axis=1)
        print(temp_1.shape)
        print(class1 + ":finish!")
        act_path = os.path.join(bath_path, 'fea_select', 'candidate')
        if not os.path.exists(act_path):
            os.mkdir(act_path)
        np.save(os.path.join(act_path, str(class1) + '_act.npy'), temp_1)
        np.save(os.path.join(act_path, str(class1) + '_file_name.npy'), np.array(temp_list))
        np.save(os.path.join(act_path, str(class1) + '_fea_name.npy'), np.array(temp_index))


if __name__ == '__main__':

    # ① 从 E:/dataset/dataset_2/fea 文件夹下手动筛选特征

    # ② 计算不同特征集合在dnn中间层的输出

    model_name = 'vgg_19'
    bath_path = 'E:/dataset/dataset_2'
    model = load_model('E:/pycharmproject/pythonProject/pythonProject/DeepRF/model/vgg_19/vgg_19.h5')
    load_imagenet_act(bath_path, model, model_name)
