import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


class VGG16Block(layers.Layer):
    def __init__(self, output_channels, l2_reg):
        super(VGG16Block, self).__init__()
        self.conv = layers.Conv2D(
            filters=output_channels,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            bias_regularizer=tf.keras.regularizers.l2(l2_reg),
        )
        self.bn = layers.BatchNormalization()

    def __call__(self, inputs):
        return self.bn(self.conv(inputs))


def vgg(
    input_shape,
    activation="relu",
    num_classes=10,
    fc1_units=200,
    fc2_units=100,
    l2_reg=0.0,
    nn_depth=5,
):
    inputs = layers.Input(shape=input_shape)
    # 64 x 64
    c1 = VGG16Block(output_channels=64, l2_reg=l2_reg)(inputs)
    c1 = layers.Dropout(0.3)(c1)
    c1 = VGG16Block(output_channels=64, l2_reg=l2_reg)(c1)
    c1 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(c1)
    # 32 x 32
    c2 = VGG16Block(output_channels=128, l2_reg=l2_reg)(c1)
    c2 = layers.Dropout(0.4)(c2)
    c2 = VGG16Block(output_channels=128, l2_reg=l2_reg)(c2)
    c2 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(c2)
    # 16 x 16
    c3 = VGG16Block(output_channels=256, l2_reg=l2_reg)(c2)
    c3 = layers.Dropout(0.4)(c3)
    c3 = VGG16Block(output_channels=256, l2_reg=l2_reg)(c3)
    c3 = layers.Dropout(0.4)(c3)
    c3 = VGG16Block(output_channels=256, l2_reg=l2_reg)(c3)
    c3 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(c3)
    # 8 x 8
    c4 = VGG16Block(output_channels=512, l2_reg=l2_reg)(c3)
    c4 = layers.Dropout(0.4)(c4)
    c4 = VGG16Block(output_channels=512, l2_reg=l2_reg)(c4)
    c4 = layers.Dropout(0.4)(c4)
    c4 = VGG16Block(output_channels=512, l2_reg=l2_reg)(c4)
    c4 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(c4)

    c5 = VGG16Block(output_channels=512, l2_reg=l2_reg)(c4)
    c5 = layers.Dropout(0.4)(c5)
    c5 = VGG16Block(output_channels=512, l2_reg=l2_reg)(c5)
    c5 = layers.Dropout(0.4)(c5)
    c5 = VGG16Block(output_channels=512, l2_reg=l2_reg)(c5)
    c5 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(c5)

    if nn_depth == 2:
        features = layers.Flatten()(c2)
    elif nn_depth == 5:
        features = layers.Flatten()(c5)
    else:
        raise RuntimeError(
            "For now, VGG only accepts nn_depth==2 or nn_depth==5. nn_depth=={} was provided".format(
                nn_depth
            )
        )

    d1 = layers.Dense(
        fc1_units,
        activation="relu",
        name="fc_head_1",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(features)
    d1 = layers.Dense(
        fc2_units,
        activation=activation,
        name="fc_head_2",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(d1)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(d1)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer(),
        metrics=["acc"],
    )

    return model
