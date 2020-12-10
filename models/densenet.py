import tensorflow as tf
from typing import Tuple


def densenet121(num_classes: int, input_shape: Tuple[int, int, int], depth: int):
    assert depth == 121
    input_tensor = tf.keras.layers.Input(input_shape)

    X = input_tensor

    if input_shape[2] == 1:

        def resize_fn(image):
            import tensorflow as tf

            return tf.image.grayscale_to_rgb(image)

        rgb_shape = (input_shape[0], input_shape[1], 3)
        X = tf.keras.layers.Lambda(resize_fn, output_shape=rgb_shape)(input_tensor)
        input_shape = rgb_shape

    print(f"input shape: {input_shape}")

    densenet_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    X = densenet_model(X)
    X = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
    )(X)

    model = tf.keras.Model(inputs=input_tensor, outputs=X)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["acc"],
    )
    return model
