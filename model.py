import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, BatchNormalization, AveragePooling2D

from utils import INPUT_SHAPE, batch_generator

data_dir = 'dataset'

def load_data():
    """
    Load training data and split it into training & validation set
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


def build_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))

    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))

    model.add(Dense(units=50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))

    model.add(Dense(units=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))

    model.add(Dense(units=1))

    return model


def train_model(model, X_train, X_valid, y_train, y_valid):
    num_of_epoch = 10
    batch_size = 32
    samples_per_epoch = 1000


    checkpoint = ModelCheckpoint('models/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit_generator(batch_generator('dataset', X_train, y_train, batch_size, True),
                                  steps_per_epoch= samples_per_epoch,
                                  epochs=num_of_epoch,
                                  validation_data=batch_generator('dataset', X_valid, y_valid, batch_size, True),
                                  validation_steps=len(X_valid) // batch_size,
                                  callbacks=[checkpoint],
                                  verbose=1)
    return history
