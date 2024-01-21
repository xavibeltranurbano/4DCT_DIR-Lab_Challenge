# -----------------------------------------------------------------------------
# ResUnet file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

from .network import Network
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras import Model, Input

class ResUnet(Network):
    def __init__(self, img_rows=256, img_cols=256, channels=3, classes=1):
        super().__init__(img_rows, img_cols, channels, classes)

    def get_model(self):
        n_layers = 3
        inputs = Input((self.img_rows, self.img_cols, self.channels))

        conv1 = Conv2D(32, (3, 3), activation=None, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        res1 = self.residual_block(32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(res1)

        res2 = self.residual_block(64, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(res2)

        res3 = self.residual_block(128, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(res3)

        res4 = self.residual_block(256, pool3)

        up5 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(res4), res3], axis=3)
        ures1 = self.residual_block(128, up5)

        up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(ures1), res2], axis=3)
        ures2 = self.residual_block(64, up6)

        up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(ures2), res1], axis=3)
        ures3 = self.residual_block(32, up7)

        if self.classes==1:
            conv_output = Conv2D(1, (1, 1), activation='sigmoid')(ures3)
        else:
            conv_output = Conv2D(self.classes, (1, 1), activation='softmax')(ures3)
        model = Model(inputs=inputs, outputs=conv_output)
        return model