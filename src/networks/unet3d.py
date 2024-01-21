# -----------------------------------------------------------------------------
# Unet 2D file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

from tensorflow.keras.layers import BatchNormalization, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate
from tensorflow.keras import Model, Input

class Unet3D(Model):
    def __init__(self, img_depth=256, img_rows=256, img_cols=256, channels=1, classes=1):
        self.img_depth = img_depth
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.classes = classes
        
    def get_model(self):
        inputs = Input((self.img_depth, self.img_rows, self.img_cols, self.channels))

        factor=1
        conv1 = Conv3D(32*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(32*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv3D(64*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(128*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv3D(256*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        
        up5 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(conv4), conv3], axis=4)
        conv5 = Conv3D(128*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv3D(128*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)

        up6 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(conv5), conv2], axis=4)
        conv6 = Conv3D(64*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv3D(64*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(conv6), conv1], axis=4)
        conv7 = Conv3D(32*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv3D(32*factor, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)
        if self.classes==1:
            conv_output = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
        else:
            conv_output = Conv3D(self.classes, (1, 1, 1), activation='softmax')(conv7)
        model = Model(inputs=inputs, outputs=conv_output)
        return model
