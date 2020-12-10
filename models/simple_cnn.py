import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def simple_cnn(input_shape, num_classes=10):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(
        filters=16, kernel_size=7, strides=1, padding="SAME", activation="relu"
    )(inputs)
    c1 = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(c1)
    c2 = layers.Conv2D(
        filters=32, kernel_size=5, strides=1, padding="SAME", activation="relu"
    )(c1)
    c2 = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(c2)

    features = layers.Flatten()(c2)
    # No activation
    d1 = layers.Dense(100, activation="relu")(features)

    outputs = layers.Dense(num_classes, activation="softmax")(d1)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer(),
        metrics=["acc"],
    )
    return model
