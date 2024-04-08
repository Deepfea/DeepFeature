import keras.applications.vgg16
import tensorflow as tf
import keras
from keras import Sequential, regularizers
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.models import load_model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Reshape, BatchNormalization
import os
from keras.optimizers import RMSprop, SGD
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

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

def vgg16():
    input_tensor = Input(shape=[32, 32, 3])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x, name='vgg16')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # batch_size = 128  # orig paper trained all networks with batch_size=128
    # epochs = 32
    # num_classes = 10
    #
    # dataset_path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_2/dataset'
    # x_train, y_train, x_test, y_test = load_data(dataset_path)
    # print("data loaded!!!")
    #
    # # Input image dimensions.
    # input_shape = x_train.shape[1:]
    # print(input_shape)
    #
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    # print('y_train shape:', y_train.shape)
    #
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)


    # model = vgg16()
    # checkpoint = ModelCheckpoint(filepath='/media/usr/external/home/usr/project/deepfeature_data/dataset_2/vgg16/model/vgg16.h5',
    #                              monitor='val_accuracy', mode='auto', save_best_only='True')
    # model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[checkpoint])
    # path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_2/vgg16/model/vgg16.h5'
    # m = load_model(path)
    # loss, acc = m.evaluate(x_test, y_test, batch_size=32)
    # print(acc)

    # model = load_model(path)
    # loss, acc = model.evaluate(x_train, y_train, batch_size=128)
    # print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    pass

