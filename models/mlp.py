import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def mlp(input_shape, activation="relu", num_classes=10, l2_reg=0.0, num_layers=1):
    inputs = layers.Input(shape=input_shape)
    features = layers.Flatten()(inputs)
    d1 = layers.Dense(
        100,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(features)

    for i in range(num_layers - 1):
        d1 = layers.Dense(
            100,
            activation="relu",
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
