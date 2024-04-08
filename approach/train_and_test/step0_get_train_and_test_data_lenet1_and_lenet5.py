import os
import numpy as np
import skimage.io as io
from PIL import Image

def get_mnist_pic(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return x_train, y_train, x_test, y_test

def save_pic(x_train, y_train, x_test, y_test, pic_save_path):
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)
    period_list = ['train', 'test']
    for period_num in range(len(period_list)):
        period_save_path = os.path.join(pic_save_path, period_list[period_num])
        if not os.path.exists(period_save_path):
            os.mkdir(period_save_path)
        pic_period_save_path = os.path.join(period_save_path, 'pic')
        if not os.path.exists(pic_period_save_path):
            os.mkdir(pic_period_save_path)
        cla = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(10):
            class_save_name = os.path.join(pic_period_save_path, str(i))
            if not os.path.exists(class_save_name):
                os.mkdir(class_save_name)
        if period_num == 0:
            x = x_train
            y = y_train
        else:
            x = x_test
            y = y_test
        for i in range(len(x)):
            file_name = os.path.join(pic_period_save_path, str(y[i]), str(y[i]) + '_' + str(cla[y[i]])+'.jpg')
            im = Image.fromarray(x[i])
            im.save(file_name)
            cla[y[i]] += 1


def pic_to_npy(dataset_path):
    period_list = ['train', 'test']
    for period in period_list:
        print(period)
        period_path = os.path.join(dataset_path, period)
        for class_num in range(10):
            class_pic_npy = []
            class_pic_name = []
            class_label_name = []
            class_path = os.path.join(period_path, 'pic', str(class_num))
            file = os.listdir(class_path)
            for file_num in range(len(file)):
                load_pic_path = os.path.join(class_path, str(class_num) + '_' + str(file_num) + '.jpg')
                pic = io.imread(load_pic_path)
                class_pic_npy.append(pic)
                class_pic_name.append(str(class_num) + '_' + str(file_num) + '.jpg')
                class_label_name.append(class_num)
            class_pic_name = np.array(class_pic_name)
            print(class_pic_name)
            class_pic_npy = np.array(class_pic_npy)
            print(class_pic_npy.shape)
            class_pic_npy = class_pic_npy.reshape(len(class_pic_npy), 28, 28, 1)
            print(class_pic_npy.shape)
            class_pic_npy = class_pic_npy.astype('float32') / 255.0
            # print(class_pic_npy[0])
            class_label_name = np.array(class_label_name)
            print(class_label_name)
            npy_path = os.path.join(period_path, 'npy')
            if not os.path.exists(npy_path):
                os.mkdir(npy_path)
            np.save(os.path.join(npy_path, str(class_num) + '_' + period + '_x.npy'), class_pic_npy)
            np.save(os.path.join(npy_path, str(class_num) + '_' + period + '_y.npy'), class_label_name)
            np.save(os.path.join(npy_path, str(class_num) + '_' + period + '_name.npy'), class_pic_name)

if __name__ == "__main__":

    pass





