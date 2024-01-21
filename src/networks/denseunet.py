# -----------------------------------------------------------------------------
# Dense Unet file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------


from .network import Network
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras import Model, Input

class DenseUnet(Network):
    def __init__(self, img_rows=256, img_cols=256, channels=3, classes=1):
        super().__init__(img_rows, img_cols, channels, classes)

    def get_model(self):
        n_layers = 15
        inputs = Input((self.img_rows, self.img_cols, self.channels))

        conv1 = Conv2D(32, (3, 3), activation=None, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        dense1 = self.DenseBlock2D(32, conv1, n_layers)
        pool1 = MaxPooling2D(pool_size=(2, 2))(dense1)

        dense2 = self.DenseBlock2D(64, pool1, n_layers)
        pool2 = MaxPooling2D(pool_size=(2, 2))(dense2)

        dense3 = self.DenseBlock2D(128, pool2, n_layers)
        pool3 = MaxPooling2D(pool_size=(2, 2))(dense3)

        dense4 = self.DenseBlock2D(256, pool3, n_layers)

        up5 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(dense4), dense3], axis=3)
        uconv1 = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer='he_normal')(up5)
        uconv1 = BatchNormalization()(uconv1)
        uconv1 = Activation('relu')(uconv1)

        up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(uconv1), dense2], axis=3)
        uconv2 = Conv2D(64, (3, 3), activation=None, padding='same', kernel_initializer='he_normal')(up6)
        uconv2 = BatchNormalization()(uconv2)
        uconv2 = Activation('relu')(uconv2)

        up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(uconv2), dense1], axis=3)
        uconv3 = Conv2D(32, (3, 3), activation=None, padding='same', kernel_initializer='he_normal')(up7)
        uconv3 = BatchNormalization()(uconv3)
        uconv3 = Activation('relu')(uconv3)

        if self.classes==1:
            conv_output = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
        else:
            conv_output = Conv2D(self.classes, (1, 1), activation='softmax')(conv7)
        model = Model(inputs=inputs, outputs=conv_output)
        return model
