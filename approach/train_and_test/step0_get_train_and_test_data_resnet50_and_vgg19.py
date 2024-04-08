import os

import numpy as np
import skimage.io as io



def save_test_pic(path):
    for class_num in range(10):
        name_path = os.path.join(path, 'dataset', 'train_and_test_img_name', 'test_name', str(class_num) + '_test_name.npy')
        name_npy = np.load(name_path)
        for name_num in range(len(name_npy)):
            load_pic_path = os.path.join(path, 'image_new', str(class_num), name_npy[name_num])
            img = io.imread(load_pic_path)
            save_class_path = os.path.join(path, 'dataset', 'test', 'pic', str(class_num))
            if not os.path.exists(save_class_path):
                os.makedirs(save_class_path)
            save_pic_path = os.path.join(save_class_path, str(class_num) + '_' + str(name_num) + '.jpg')
            io.imsave(save_pic_path, img)

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
            class_pic_npy = class_pic_npy.astype('float32') / 255.0
            print(class_pic_npy[0])
            class_label_name = np.array(class_label_name)
            print(class_label_name)
            np.save(os.path.join(period_path, 'npy', str(class_num) + '_' + period + '_x.npy'), class_pic_npy)
            np.save(os.path.join(period_path, 'npy', str(class_num) + '_' + period + '_y.npy'), class_label_name)
            np.save(os.path.join(period_path, 'npy', str(class_num) + '_' + period + '_name.npy'), class_pic_name)

if __name__ == "__main__":
    pass





