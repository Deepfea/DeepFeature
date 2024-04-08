import os
import re
import keras.applications.vgg19
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Reshape, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.models import load_model

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

def vgg19():
    input_tensor = Input(shape=[224, 224, 3])
    # input_tensor = Input(shape=[32, 32, 3])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    # x = Dense(1000, activation='softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x, name='vgg19')
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # dataset_path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_3/dataset'
    # train_x, train_y, test_x, test_y = load_data(dataset_path)
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    #
    # train_y = keras.utils.to_categorical(train_y, 10)
    # test_y = keras.utils.to_categorical(test_y, 10)
    # print("data loaded!!!")
    #
    # model_path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_3/vgg19/model/vgg19.h5'
    # model_path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_3/vgg19/11_select_data_retrain/cw/retrain_model/0_vgg19.h5'
    # model = vgg19()
    # checkpoint = ModelCheckpoint(filepath=model_path,
    #                              monitor='val_accuracy', mode='auto', save_best_only='True')
    # model.fit(train_x, train_y, epochs=32, batch_size=64, validation_data=(test_x, test_y), callbacks=[checkpoint])

    # model = keras.models.load_model(model_path)
    # loss, acc = model.evaluate(train_x, train_y, batch_size=200)
    # print('model accurancy: {:5.2f}%'.format(100 * acc))
    # print('model, loss: ' + str(loss))
    # model = load_model(model_path)
    # loss, acc = model.evaluate(train_x, train_y, batch_size=128)
    # print('Normal model accurancy: {:5.2f}%'.format(100 * acc))

    pass