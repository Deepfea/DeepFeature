from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import pandas as pd
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

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# resnet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
# n = 3




def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):

    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                  padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    x = Dense(10, name='before_softmax')(y)
    outputs = Activation('softmax', name='predictions')(x)
    # outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
if __name__ == '__main__':
    # Training parameters
    # batch_size = 64  # orig paper trained all networks with batch_size=128
    # epochs = 32
    # num_classes = 10
    # version = 1
    # depth = 50
    # model_type = 'ResNet%dv%d' % (depth, version)
    # print(model_type)
    #
    # dataset_path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_3/dataset'
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
    #
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # model = resnet_v1(input_shape=input_shape, depth=depth)
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
    # filepath = '/media/usr/external/home/usr/project/deepfeature_data/dataset_3/resnet50/resnet50.h5'
    # # # Prepare callbacks for model saving and for learning rate adjustment.
    # checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    # callbacks = [checkpoint, lr_reducer, lr_scheduler]
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
    #               shuffle=True, callbacks=callbacks)

    # model = keras.models.load_model('/media/usr/external/home/usr/project/deepfeature_data/dataset_3/resnet50/model/resnet50.h5')
    # loss, acc = model.evaluate(x_test, y_test, batch_size=128)
    # print('model accurancy: {:5.2f}%'.format(100 * acc))
    # print('model, loss: ' + str(loss))
    # path = '/media/usr/external/home/usr/project/deepfeature_data/dataset_3/resnet50/model/resnet50.h5'
    # model = load_model(path)
    # loss, acc = model.evaluate(x_train, y_train, batch_size=128)
    # print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    pass
