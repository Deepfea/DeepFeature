import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation
import os
import skimage.io
from keras.models import load_model

def Lenet5():
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    input_tensor = Input(shape=[28, 28, 1])
    # block1
    x = Conv2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # block2
    x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x)
    return model

def load_data(dataset_path):
    period_list = ['train', 'test']
    for period in period_list:
        period_path = os.path.join(dataset_path, period)
        for class_num in range(10):
            temp_x = np.load(os.path.join(period_path, 'npy', str(class_num) + '_' + period + '_x.npy'))
            temp_y = np.load(os.path.join(period_path, 'npy', str(class_num) + '_' + period + '_y.npy'))
            if period == 'train':
                if class_num == 0:
                    train_x = temp_x
                    train_y = temp_y
                else:
                    train_x = np.concatenate((train_x, temp_x), axis=0)
                    train_y = np.concatenate((train_y, temp_y), axis=0)
            else:
                if class_num == 0:
                    test_x = temp_x
                    test_y = temp_y
                else:
                    test_x = np.concatenate((test_x, temp_x), axis=0)
                    test_y = np.concatenate((test_y, temp_y), axis=0)
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    # batch_size = 128
    # epochs = 32
    # num_classes = 10
    #
    #
    # dataset_path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_1/dataset'
    # x_train, y_train, x_test, y_test = load_data(dataset_path)
    # print("data loaded!!!")
    # input_shape = x_train.shape[1:]
    # print(input_shape)
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    # print('y_train shape:', y_train.shape)
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_1/lenet5/model/lenet5.h5'
    # model = Lenet5()
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint = ModelCheckpoint(filepath=path,
    #                              monitor='val_accuracy', mode='auto', save_best_only='True')
    # model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[checkpoint])

    # loss, acc = model.evaluate(x_test, y_test, batch_size=128)
    # print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    # model = load_model(path)
    # loss, acc = model.evaluate(x_train, y_train, batch_size=128)
    # print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    pass





