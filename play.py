import cv2 as cv
import math
import os
from random import shuffle
from tqdm import tqdm
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model


min_w = 32
min_h = 32


def pre_process(img):
    resized = cv.resize(img, (min_w, min_h))
    return resized


def get_label(path):
    if 'nolights' not in path:
        return np.array([1, 0])
    else:
        return np.array([0, 1])
    return np.array([0, 1])


def train_data_with_label(train_data, train_images=[]):
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv.imread(path)
        w, h, _ = img.shape
        if w < min_w or h < min_h:
            continue
        img = pre_process(img)
        label = get_label(train_data)
        train_images.append([np.array(img), label])
    shuffle(train_images)
    return train_images


def train_model(w, h, tr_img_data, tr_lbl_data):
    model = Sequential()

    model.add(InputLayer(input_shape=[w, h, 3]))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1,
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=50, kernel_size=5, strides=1,
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=80, kernel_size=5, strides=1,
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, input_shape=[w, h, 3], activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, input_shape=[w, h, 3], activation='softmax'))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=tr_img_data, y=tr_lbl_data, epochs=10, batch_size=10)
    model.summary()
    return model


def train(videoName):
    train_data = train_data_with_label('./train/%s/lights' % videoName)
    train_data = train_data_with_label(
        './train/%s/nolights' % videoName, train_data)

    tr_img_data = np.array([i[0] for i in train_data])
    tr_lbl_data = np.array([i[1] for i in train_data])
    model = train_model(min_w, min_h, tr_img_data, tr_lbl_data)
    return model


def save_lights(rects, videoName, frame):
    imagePath = './images/lights/%s' % videoName
    if not os.path.exists(imagePath):
        os.makedirs(imagePath)
    for cnt, img in enumerate(rects):
        cv.imwrite(imagePath + '/frame%d_%d.jpg' % (frame, cnt), img)


def predict(model, light):
    if light.shape[0] < min_w or light.shape[1] < min_h:
        return False
    data = pre_process(light)
    data = np.array(data).reshape(-1, min_w, min_h, 3)
    model_out = model.predict([data])
    if np.argmax(model_out) == 1:
        return False
    else:
        return True


def find_lghts(light, black, org, cr_h, cr_w, model):
    contours_light, _ = cv.findContours(
        light, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_black, _ = cv.findContours(
        black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    thresh_height = cr_h/2
    thresh_width = cr_w/4
    trafic_lights = []
    rects = []
    for _, c in enumerate(contours_black):
        area_black = cv.contourArea(c)
        if area_black > 0:
            # print(area_black)
            x, y, w, h = cv.boundingRect(c)
            if w < thresh_width and h < thresh_height:
                for _, cl in enumerate(contours_light):
                    area_light = cv.contourArea(cl)
                    if area_light > 0 and area_black > area_light:
                        xl, yl, wl, hl = cv.boundingRect(cl)
                        if (x < xl and x+w > xl + wl
                                and y < yl and y+h > yl+hl):
                            light = org[y:y+h, x:x+w]
                            if predict(model, light):
                                trafic_lights.append(c)
                                rects.append(light)
                                break
    output = org.copy()
    for c in trafic_lights:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return (output, rects)


def draw_frame(img, videoName, frame, model):
    h, w, _ = img.shape
    cropped_img = img.copy()
    cropped_img = img[math.floor(h/4):math.floor(h*3/4)]
    cropped_img = cropped_img[:, math.floor(w/8):math.floor(w/2)]
    cr_h, cr_w, _ = cropped_img.shape

    gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    _, gb = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    _, black_shape = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV)
    output, rects = find_lghts(gb, black_shape, cropped_img, cr_h, cr_w, model)
    save_lights(rects, videoName, frame)
    return output


videoName = 'video-2'
model = train(videoName)
source = './videos/%s.mp4' % videoName
vidcap = cv.VideoCapture(source)
frame = 0

while True:
    success, image = vidcap.read()

    if not success:
        vidcap = cv.VideoCapture(source)
        frame = 0
        continue

    cv.imshow('video', draw_frame(image, videoName, frame, model))
    frame = frame + 1

    key = cv.waitKey(25)
    if key == 27:
        break

cv.destroyAllWindows()
