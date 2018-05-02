# Copyright (c) 2015 Lightricks. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import Model, optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, concatenate
from skimage.morphology import disk

HIGHT = 256
WIDTH = 256
NUM_CLASSES = 2
TRAIN_SIZE = 20
TEST_SIZE = 5
EPOCHS = 5
BATCH_SIZE = 1

def create_dataset(train_size):
    train_labels = np.zeros((train_size, HIGHT, WIDTH, 1))
    train_labels[0:2:9, :, 0:int(WIDTH / 2), :] = 1
    train_labels[1:2:9, 0:int(HIGHT / 2), :, :] = 1
    # noise = 0.2 * np.random.randn(train_size, HIGHT, WIDTH, 1)
    train_data = train_labels

    test_labels = np.zeros((3, HIGHT, WIDTH, 1))

    test_labels[0, 30:130, 50:240, :] = 1
    disk_element = disk(radius=100)
    test_labels[1, 30:30 + disk_element.shape[0], 24:24 + disk_element.shape[1], 0] = disk_element
    disk_element = disk(radius=82)
    test_labels[2, 70:70 + disk_element.shape[0], 70:70 + disk_element.shape[1], 0] = disk_element
    noise = 0.2 * np.random.randn(3, HIGHT, WIDTH, 1)
    test_data = test_labels + noise
    return train_data, train_labels, test_data, test_labels


def create_circles_dataset(train_size, test_size):
    data_size = train_size + test_size
    labels = np.zeros((data_size, HIGHT, WIDTH, 1))
    for i in range(data_size):
        num_circles = np.random.randint(low=4, high=8)
        for j in range(num_circles):
            radius = np.random.randint(low=10, high=50)
            disk_element = disk(radius=radius)
            (disk_r, disk_c) = disk_element.shape
            row = np.random.randint(low=0, high=HIGHT)
            col = np.random.randint(low=0, high=WIDTH)
            row_end = min(row + disk_r, HIGHT)
            col_end = min(col + disk_c, WIDTH)
            labels[i, row:row_end, col:col_end, 0] += disk_element[0:(row_end - row), 0:(col_end - col)]
    labels[labels > 1] = 1
    noise = 0.2 * np.random.randn(data_size, HIGHT, WIDTH, 1)
    data = labels + noise
    # data = labels
    data[data < 0] = 0
    data[data > 1] = 1
    # for i in range(data_size):
    #     plt.figure()
    #     plt.imshow(data[i, :, :, 0], cmap='gray')
    #     plt.title('data, i=' + str(i))
    #     plt.figure()
    #     plt.imshow(labels[i, :, :, 0], cmap='gray')
    #     plt.title('labels, i=' + str(i))
    #     plt.show()
    train_data = data[0:train_size, :, :, :]
    train_labels = labels[0:train_size, :, :, :]
    test_data = data[train_size:data_size, :, :, :]
    test_labels = labels[train_size:data_size, :, :, :]
    return train_data, train_labels, test_data, test_labels



# Down 1
input_im = Input(shape=(HIGHT, WIDTH, 1))
conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_im)
conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_1)

# Down 2
max_pool_1 = MaxPooling2D((2, 2), strides=(2, 2))(conv_1)
conv_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(max_pool_1)
conv_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_2)

# Down 3
max_pool_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)
conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(max_pool_2)
conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_3)

# Down 4
max_pool_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv_3)
conv_4 = Conv2D(512, (3, 3), padding='same', activation='relu')(max_pool_3)
conv_4 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_4)

# Bottom
max_pool_4 = MaxPooling2D((2, 2), strides=(2, 2))(conv_4)
bottom = Conv2D(1024, (3, 3), padding='same', activation='relu')(max_pool_4)
bottom = Conv2D(1024, (3, 3), padding='same', activation='relu')(bottom)

# Up 4
up_sampled_4 = UpSampling2D(size=(2, 2))(bottom)
up_sampled_4 = Conv2D(512, (2, 2), padding='same', activation='relu')(up_sampled_4)
concat_4 = concatenate([conv_4, up_sampled_4], axis=3)
# now input has 1024 features
up_conv_4 = Conv2D(512, (3, 3), padding='same', activation='relu')(concat_4)
up_conv_4 = Conv2D(512, (3, 3), padding='same', activation='relu')(up_conv_4)

# Up 3
up_sampled_3 = UpSampling2D(size=(2, 2))(up_conv_4)
up_sampled_3 = Conv2D(256, (2, 2), padding='same', activation='relu')(up_sampled_3)
concat_3 = concatenate([conv_3, up_sampled_3], axis=3)
# now input has 512 features
up_conv_3 = Conv2D(256, (3, 3), padding='same', activation='relu')(concat_3)
up_conv_3 = Conv2D(256, (3, 3), padding='same', activation='relu')(up_conv_3)

# Up 2
up_sampled_2 = UpSampling2D(size=(2, 2))(up_conv_3)
up_sampled_2 = Conv2D(128, (2, 2), padding='same', activation='relu')(up_sampled_2)
concat_2 = concatenate([conv_2, up_sampled_2], axis=3)
# now input has 256 features
up_conv_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(concat_2)
up_conv_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(up_conv_2)

# Up 1
up_sampled_1 = UpSampling2D(size=(2, 2))(up_conv_2)
up_sampled_1 = Conv2D(64, (2, 2), padding='same', activation='relu')(up_sampled_1)
concat_1 = concatenate([conv_1, up_sampled_1], axis=3)
# now input has 256 features
up_conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(concat_1)
up_conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(up_conv_1)

output = Conv2D(2, (1, 1), padding='same', activation='softmax')(up_conv_1)

model = Model(inputs=input_im, outputs=output)

sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    # metrics=['accuracy']
)


(train_data, train_labels, test_data, test_labels) = create_circles_dataset(TRAIN_SIZE, TEST_SIZE)

one_hot_train_labels = keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
one_hot_test_labels = keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

model.fit(x=train_data, y=one_hot_train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
score = model.evaluate(test_data, one_hot_test_labels, batch_size=BATCH_SIZE)
print('score =', score)
prediction = model.predict(test_data, batch_size=BATCH_SIZE)
prediction = np.argmax(prediction, axis=3)

for i in range(TEST_SIZE):
    plt.figure()
    plt.imshow(prediction[i, :, :], cmap='gray')
    plt.title('prediction, i =' + str(i))
    plt.figure()
    plt.imshow(test_labels[i, :, :, 0], cmap='gray')
    plt.title('labels, i =' + str(i))
    plt.figure()
    plt.imshow(test_data[i, :, :, 0], cmap='gray')
    plt.title('data, i =' + str(i))
    plt.show()
