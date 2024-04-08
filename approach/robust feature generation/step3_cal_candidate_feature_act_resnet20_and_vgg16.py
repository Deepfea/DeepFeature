import os
import numpy as np
import skimage.io as io
from keras.models import load_model
from keras import Model
from skimage.transform import resize

def load_act(load_path, save_path, model_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = load_model(model_path)
    for class_num in range(10):
        load_class_path = os.path.join(load_path, str(class_num))
        fea_list = os.listdir(load_class_path)
        temp_list = []
        temp_index = []
        flag = 0
        for fea in fea_list:
            print(fea + ':')
            fea_path = os.path.join(load_class_path, fea)
            file_list = os.listdir(fea_path)
            key = 0
            for file in file_list:
                # print(file)
                file_path = os.path.join(fea_path, file)
                # print(file_path)
                img1 = io.imread(file_path)
                img = resize(img1, (32, 32, 3))
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
        if model_name == 'vgg16':
            i = Model(inputs=model.layers[0].output, outputs=model.layers[17].output)
            temp_1 = i.predict(temp_arr).reshape(len(temp_arr), -1, i.output.shape[-1])
            temp_1 = np.mean(temp_1, axis=1)
        if model_name == 'resnet20':
            i = Model(inputs=model.layers[0].output, outputs=model.layers[68].output)
            temp_1 = i.predict(temp_arr).reshape(len(temp_arr), -1, i.output.shape[-1])
            temp_1 = np.mean(temp_1, axis=1)
        print(temp_1.shape)
        print(str(class_num) + ":finish!")

        np.save(os.path.join(save_path, str(class_num) + '_act.npy'), temp_1)
        np.save(os.path.join(save_path, str(class_num) + '_file_name.npy'), np.array(temp_list))
        np.save(os.path.join(save_path, str(class_num) + '_fea_name.npy'), np.array(temp_index))


if __name__ == '__main__':

    pass
