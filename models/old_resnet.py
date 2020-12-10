from classification_models.tfkeras import Classifiers
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def resnet(input_shape, num_classes=10, fc_units=256, l2_reg=0.0, nn_depth=18):
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg",
        classes=num_classes,
    )
    features = resnet.output

    fc = layers.Dense(
        fc_units,
        activation="relu",
        name="fc_head",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(features)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(fc)

    model = Model(inputs=resnet.input, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer(),
        metrics=["acc"],
    )

    return model


def OLD_resnet(input_shape, num_classes=10, fc_units=256, l2_reg=0.0, nn_depth=18):
    ResNetModel, preprocess_input = Classifiers.get("resnet{}".format(nn_depth))
    resnet = ResNetModel(input_shape, include_top=False)
    features = layers.GlobalAveragePooling2D()(resnet.output)

    fc = layers.Dense(
        fc_units,
        activation="relu",
        name="fc_head",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(features)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(fc)

    model = Model(inputs=resnet.input, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer(),
        metrics=["acc"],
    )

    return model
