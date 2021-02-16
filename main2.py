import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.initializers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten

from keras.applications import MobileNetV2, MobileNet
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from keras.applications.mobilenet import preprocess_input
import numpy as np
import pandas as pd
import seaborn as sn
# from IPython.display import Image
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
from imutils import paths
from PIL import Image
from skimage.transform import rescale
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import cv2
from PIL import ImageFilter
from sklearn.metrics import ConfusionMatrixDisplay


# model = Sequential()
# model.add(Conv2D(8, (3, 3), padding="same", input_shape=(208, 156, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(BatchNormalization())
# model.add(Conv2D(16, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Conv2D(32, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(12))
# model.add(Activation("softmax"))

seed = random.randint(1, 1000)

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(208, 156, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dropout(rate = .2)(x)
x = BatchNormalization()(x)
x = Dense(1280, activation='relu',  kernel_initializer=glorot_uniform(seed), bias_initializer='zeros')(x)
# x = Dropout(rate = .2)(x)
x = BatchNormalization()(x)
preds = Dense(12, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

model = Model(inputs = base_model.input, outputs = preds)

print("[INFO] loading images...")
imagePaths = paths.list_images("Dataset_GMB")
data = []
labels = []
for imagePath in imagePaths:
    image = Image.open(imagePath)
    # image = cv2.imread(imagePath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, dsize=(208,156), interpolation=cv2.INTER_CUBIC)
    image = np.array(image.resize((208, 156)))
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.EDGE_ENHANCE)
    blur = image.filter(ImageFilter.GaussianBlur(radius=0.2))
    np_blur = np.array(blur)
    np_orin = np.array(image)
    np_diff = np_blur - np_orin
    np_add_img = np_diff + np_orin
    add_img = Image.fromarray(np_add_img)
    # plt.imshow(add_img)
    # plt.show()
    # exit()
    image = np.array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.20)
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

opt = Adam(lr=1e-3, decay=1e-3)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=.0001,
                          patience=10,
                          verbose=0,
                          mode='auto',
                          baseline=None,
                          restore_best_weights=True)

checkpoint = ModelCheckpoint('GMB_classifier_efficiennet.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose=1,
                             mode='auto',
                             save_weights_only=False,
                             period=1)

print("[INFO] training network...")
opt = Adam(lr=1e-3, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32, callbacks=[earlystop, checkpoint])


model.save('model_efficiennet')

print(H.history.keys())
# summarize history for accuracy
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('Accuracy_model_mobilenet.png')
plt.show()
# summarize history for loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('loss_model_mobilenet.png')
plt.show()

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


ConfusionMatrixDisplay(testY, predictions, labels=lb.classes_)

# print('Confusion Matrix')
# cm = confusion_matrix(testY, predictions)
# df = pd.DataFrame(cm, columns=test_generator.class_indices)
# plt.figure(figsize=(80,80))
# sn.heatmap(df, annot=True)
# # plt.savefig('Confussion_Matrix_MobileNet.png')
# plt.show()
# import tensorflow as tf

# saved_model_dir = '/content/TFLite'
# # tf.saved_model.save(model, saved_model_dir)
# tf.keras.models.save_model(model, saved_model_dir)
# # tf.keras.models.save_model(model, saved_model_dir)
#
# converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
#
# # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# # converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
# # converter = tf.lite.TFLiteConverter.from_keras_model(saved_model_dir)
# # converter.experimental_new_converter = True
# # tflite_model = converter.convert()
# # tflite_model = converter.convert()
#
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)
