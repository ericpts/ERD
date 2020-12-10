import functools
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

keras = tf.keras


def _resnet_layer(
    inputs,
    num_filters,
    l2_reg,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_norm=True,
):
    """2D Convolution-Batch Normalization-Activation stack builder.
    Args:
      inputs (tensor): input tensor from input image or previous layer
      num_filters (int): Conv2D number of filters
      kernel_size (int): Conv2D square kernel dimensions
      strides (int): Conv2D square stride dimensions
      activation (string): Activation function string.
      depth (int): ResNet depth; used for initialization scale.
      batch_norm (bool): whether to include batch normalization
      std_prior_scale (float): Scale for log-normal hyperprior.
      eb_prior_fn (callable): Empirical Bayes prior for use with TFP layers.
      examples_per_epoch (int): Number of examples per epoch for variational KL.
    Returns:
        x (tensor): tensor as input to the next layer
    """
    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        use_bias=False,
    )

    x = inputs
    x = conv(x)
    x = keras.layers.BatchNormalization()(x) if batch_norm else x
    x = keras.layers.Activation(activation)(x) if activation is not None else x
    x = x
    return x


def resnet_v1(
    input_shape,
    depth,
    num_classes,
    l2_reg,
    num_filters=64,
):
    """ResNet Version 1 Model builder [a]."""
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    # Start model definition.
    num_res_blocks = int((depth - 2) / 6) - 1

    input_layer = layers.Input(shape=input_shape)

    activation = "relu"
    resnet_layer = functools.partial(
        _resnet_layer, activation=activation, l2_reg=l2_reg
    )

    kernel_size = 3
    strides = 1
    if input_shape[0] == 224:
        kernel_size = 7
        strides = 2

    x = resnet_layer(
        inputs=input_layer,
        num_filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
    )
    # Instantiate the stack of residual units
    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_norm=True,
                )
            x = keras.layers.add([x, y])
            x = keras.layers.Activation(activation)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    features = keras.layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(features)

    model = Model(inputs=input_layer, outputs=outputs)

    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["acc"],
    )

    return model
