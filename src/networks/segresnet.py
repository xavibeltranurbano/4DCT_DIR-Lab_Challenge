# -----------------------------------------------------------------------------
# SegResNet file adapted from MONAI to tensorflow
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

from tensorflow.keras import layers, models


def get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    if spatial_dims == 2:
        return layers.Conv2D(out_channels, kernel_size, strides=stride, padding=padding)
    elif spatial_dims == 3:
        return layers.Conv3D(out_channels, kernel_size, strides=stride, padding=padding)
    else:
        raise ValueError("`spatial_dims` can only be 2 or 3.")

def get_upsample_layer(spatial_dims, out_channels, upsample_mode='nearest'):
    if spatial_dims == 2:
        return layers.UpSampling2D(size=(2, 2), interpolation=upsample_mode)
    elif spatial_dims == 3:
        return layers.UpSampling3D(size=(2, 2, 2))
    else:
        raise ValueError("`spatial_dims` can only be 2 or 3.")

def ResBlock(spatial_dims, filters, kernel_size=3, norm='batch', activation='relu'):
    block = models.Sequential()
    block.add(get_conv_layer(spatial_dims, filters, filters, kernel_size))
    if norm == 'batch':
        block.add(layers.BatchNormalization())
    elif norm == 'group':
        # Group normalization is not natively supported in Keras, consider using a custom layer or approximation
        pass
    block.add(layers.Activation(activation))
    block.add(get_conv_layer(spatial_dims, filters, filters, kernel_size))
    block.add(layers.BatchNormalization()) if norm == 'batch' else None
    block.add(layers.Activation(activation))
    return block

def SegResNet(input_shape, spatial_dims=3, init_filters=8, in_channels=1, out_channels=2, dropout_prob=None, 
              act='relu', norm='batch', blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), upsample_mode='nearest'):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution
    x = get_conv_layer(spatial_dims, in_channels, init_filters)(inputs)
    if dropout_prob is not None:
        x = layers.Dropout(dropout_prob)(x)

    # Down-sampling layers
    down_layers = []
    for i, num_blocks in enumerate(blocks_down):
        layer_in_channels = init_filters * (2 ** i)
        for _ in range(num_blocks):
            x = ResBlock(spatial_dims, layer_in_channels, norm=norm, activation=act)(x)
        down_layers.append(x)
        if i < len(blocks_down) - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x) if spatial_dims == 2 else layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Up-sampling layers
    for i, num_blocks in enumerate(blocks_up):
        x = get_upsample_layer(spatial_dims, init_filters, upsample_mode=upsample_mode)(x)
        x = layers.concatenate([x, down_layers[-(i+2)]])
        for _ in range(num_blocks):
            x = ResBlock(spatial_dims, layer_in_channels // (2 ** (i + 1)), norm=norm, activation=act)(x)

    # Final Convolution
    x = layers.Conv2D(out_channels, 1, activation='softmax')(x) if spatial_dims == 2 else layers.Conv3D(out_channels, 1, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model