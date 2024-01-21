# -----------------------------------------------------------------------------
# Super class network file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Add, Activation, Multiply, BatchNormalization


class Network:
    def __init__(self, img_rows=256, img_cols=256, channels=3, classes=4):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.classes = classes

    def conv_block(self,size, x):
        conv1 = BatchNormalization()(x)
        conv1 = Conv2D(size, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = Activation('relu')(conv1)
        return conv1

    def residual_block(self,size, x):
        res = self.conv_block(size, x)
        res = self.conv_block(size, res)

        shortcut = Conv2D(size, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        shortcut = BatchNormalization()(shortcut)
        output = Add()([shortcut, res])
        return output

    def DenseBlock2D(self,size, x, num_layers):
        # Dense block
        for _ in range(num_layers):
            conv = self.conv_block(size, x)
            x = concatenate([x, conv])
        return x

    def MultiConv2D(self,size, x):
        # MultiResolution bloc
        conv3x3 = self.conv_block(size,x)
        conv5x5 = self.conv_block(size,conv3x3)
        conv7x7 =  self.conv_block(size,conv5x5)
        conv_block = concatenate([conv3x3, conv5x5, conv7x7])
        return conv_block

    def AttentionBlock(self,inputs, binary_mask):
        # Attention Block
        attention_map = Multiply()([inputs, binary_mask])
        return attention_map
