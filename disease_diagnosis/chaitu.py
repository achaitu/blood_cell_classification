# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 08:34:42 2018

@author: jeevan
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.utils import np_utils
from keras.callbacks import CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import cv2
from keras import backend as K

num_classes = 2
epochs = 10
#BASE_DIR = 'H:/Projects/Blood_Classification/My_Work/this_sem_work/ALL_IDB_CLassification/'
batch_size = 16
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120, 160, 3), output_shape=(120, 160, 3)))
    model.add(Conv2D(32, (3, 3), input_shape=(120, 160, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model

def get_data(folder):
    X = []
    y = []

    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            for image_filename in os.listdir(folder + wbc_type):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (160,120))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(wbc_type)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y


X_train, y_train = get_data( '/ALL_IDB_2/Train_Classes/')
X_test, y_test = get_data('/ALL_IDB_2/Test_Classes/')

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)
encoded_y_test = encoder.transform(y_test)

y_train = np_utils.to_categorical(encoded_y_train)
y_test = np_utils.to_categorical(encoded_y_test)

model = get_model()
csv_logger = CSVLogger('training.log')

details = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    shuffle=True,
    batch_size=100, callbacks=[csv_logger])

model.load_weights('all_idb_classification_weights.h5')

y_pred = np.rint(model.predict(X_test))
print(accuracy_score(y_test, y_pred))
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
print(confusion_matrix(y_test,y_pred))

#get_3rd_layer_output = K.function([model.layers[0].input],
                                  #[model.layers[3].output])
#layer_output = get_3rd_layer_output([X_test])[0]

#cv2.imshow('image',layer_output[0,:,:,1])
cv2.waitKey(20000)


#for i in range(29):
    #cv2.imwrite('hidden_layer_features/im'+str(i)+'.png',layer_output[0,:,:,i:i+3])