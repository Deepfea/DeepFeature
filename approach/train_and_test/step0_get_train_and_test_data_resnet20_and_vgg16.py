import os
import numpy as np
import skimage.io as io
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def get_cifar_test_pic(path):
    batch_path = os.path.join(path, 'batch')
    meta = unpickle(os.path.join(batch_path, 'batches.meta'))
    label_name = meta[b'label_names']
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    content = unpickle(os.path.join(batch_path, 'test_batch'))
    print('load testing data...')
    print(content.keys())
    test_path = os.path.join(path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    test_path = os.path.join(test_path, 'pic')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        index_i = labels.index(label_name[content[b'labels'][j]].decode())
        class_path = os.path.join(test_path, str(index_i))
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        img_name = os.path.join(class_path, str(index_i) + '_' + str(nums[index_i]) + '.jpg')
        io.imsave(img_name, img)
        nums[index_i] += 1

def get_cifar_train_pic(path):
    batch_path = os.path.join(path, 'batch')
    meta = unpickle(os.path.join(batch_path, 'batches.meta'))
    label_name = meta[b'label_names']
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 6):
        content = unpickle(os.path.join(batch_path, 'data_batch_' + str(i)))
        print('load training data...')
        print(content.keys())
        train_path = os.path.join(path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_path = os.path.join(train_path, 'pic')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        for j in range(10000):
            img = content[b'data'][j]
            img = img.reshape(3, 32, 32)
            img = img.transpose(1, 2, 0)
            index_i = labels.index(label_name[content[b'labels'][j]].decode())
            class_path = os.path.join(train_path, str(index_i))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            img_name = os.path.join(class_path, str(index_i) + '_' + str(nums[index_i]) + '.jpg')
            io.imsave(img_name, img)
            nums[index_i] += 1


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
            npy_path = os.path.join(period_path, 'npy')
            if not os.path.exists(npy_path):
                os.makedirs(npy_path)
            np.save(os.path.join(npy_path, str(class_num) + '_' + period + '_x.npy'), class_pic_npy)
            np.save(os.path.join(npy_path, str(class_num) + '_' + period + '_y.npy'), class_label_name)
            np.save(os.path.join(npy_path, str(class_num) + '_' + period + '_name.npy'), class_pic_name)

if __name__ == "__main__":
    pass





