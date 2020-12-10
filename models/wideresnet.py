import os

import tensorflow as tf
from functools import partial
from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    Input,
    Activation,
    Add,
    Dense,
    Flatten,
)
from tensorflow.keras.regularizers import l2

WEIGHT_INIT = "he_normal"  # follows the 'MSRinit(model)' function in utils.lua


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride, l2_reg: float):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]

        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization()(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization()(net)
                    convs = Activation("relu")(convs)
                convs = Conv2D(
                    n_bottleneck_plane,
                    (v[0], v[1]),
                    strides=v[2],
                    padding=v[3],
                    kernel_initializer=WEIGHT_INIT,
                    kernel_regularizer=l2(l2_reg),
                    use_bias=False,
                )(convs)
            else:
                convs = BatchNormalization()(convs)
                convs = Activation("relu")(convs)
                convs = Conv2D(
                    n_bottleneck_plane,
                    (v[0], v[1]),
                    strides=v[2],
                    padding=v[3],
                    kernel_initializer=WEIGHT_INIT,
                    kernel_regularizer=l2(l2_reg),
                    use_bias=False,
                )(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Conv2D(
                n_output_plane,
                (1, 1),
                strides=stride,
                padding="same",
                kernel_initializer=WEIGHT_INIT,
                kernel_regularizer=l2(l2_reg),
                use_bias=False,
            )(net)
        else:
            shortcut = net

        return Add()([convs, shortcut])

    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2, int(count + 1)):
            net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
        return net

    return f


def wrn28_10(
    num_classes: int, input_shape: Tuple[int, int, int], depth: int, l2_reg: float
):
    wide_factor = 10

    assert (depth - 4) % 6 == 0
    n = (depth - 4) / 6

    inputs = Input(shape=input_shape)

    n_stages = [16, 16 * wide_factor, 32 * wide_factor, 64 * wide_factor]

    conv1 = Conv2D(
        n_stages[0],
        (3, 3),
        strides=1,
        padding="same",
        kernel_initializer=WEIGHT_INIT,
        kernel_regularizer=l2(l2_reg),
        use_bias=False,
    )(
        inputs
    )  # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = partial(_wide_basic, l2_reg=l2_reg)
    conv2 = _layer(
        block_fn,
        n_input_plane=n_stages[0],
        n_output_plane=n_stages[1],
        count=n,
        stride=(1, 1),
    )(
        conv1
    )  # "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(
        block_fn,
        n_input_plane=n_stages[1],
        n_output_plane=n_stages[2],
        count=n,
        stride=(2, 2),
    )(
        conv2
    )  # "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(
        block_fn,
        n_input_plane=n_stages[2],
        n_output_plane=n_stages[3],
        count=n,
        stride=(2, 2),
    )(
        conv3
    )  # "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization()(conv4)
    relu = Activation("relu")(batch_norm)

    # Classifier block
    pool = GlobalAveragePooling2D()(relu)
    flatten = Flatten()(pool)
    predictions = Dense(
        units=num_classes,
        kernel_initializer=WEIGHT_INIT,
        use_bias=False,
        kernel_regularizer=l2(l2_reg),
        activation="softmax",
    )(flatten)

    model = Model(inputs=inputs, outputs=predictions)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["acc"],
    )
    return model
