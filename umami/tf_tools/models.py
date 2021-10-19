"""
Implementations by Johnny Raine
"""

from umami.configuration import logger  # isort:skip

import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    GlobalMaxPool1D,
    Input,
    Lambda,
    Masking,
    TimeDistributed,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from .layers import (
    Attention,
    AttentionPooling,
    ConditionalAttention,
    ConditionalDeepSet,
    DeepSet,
    DenseNet,
    MaskedAverage1DPooling,
    Sum,
)


def Umami_model(train_config=None, input_shape=None, njet_features=None):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    if train_config.model_file is not None:
        # Load DIPS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        umami = load_model(
            train_config.model_file, {"Sum": Sum}, compile=False
        )

    else:
        logger.info("No modelfile provided! Initialize a new one!")

        # Set the track input
        trk_inputs = Input(shape=input_shape)

        # Masking the missing tracks
        masked_inputs = Masking(mask_value=0)(trk_inputs)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(NN_structure["DIPS_ppm_units"]):

            tdd = TimeDistributed(
                Dense(phi_nodes, activation="linear"), name=f"Phi{i}_Dense"
            )(tdd)

            if batch_norm:
                tdd = TimeDistributed(
                    BatchNormalization(), name=f"Phi{i}_BatchNormalization"
                )(tdd)

            if dropout != 0:
                tdd = TimeDistributed(
                    Dropout(rate=dropout), name=f"Phi{i}_Dropout"
                )(tdd)

            tdd = TimeDistributed(
                Activation(activations.relu), name=f"Phi{i}_ReLU"
            )(tdd)

        # This is where the magic happens... sum up the track features!
        F = Sum(name="Sum")(tdd)

        for j, (F_nodes, p) in enumerate(
            zip(
                NN_structure["DIPS_dense_units"],
                [dropout] * len(NN_structure["DIPS_dense_units"][:-1]) + [0],
            )
        ):

            F = Dense(F_nodes, activation="linear", name=f"F{j}_Dense")(F)
            if batch_norm:
                F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
            if dropout != 0:
                F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
            F = Activation(activations.relu, name=f"F{j}_ReLU")(F)

        dips_output = Dense(
            len(class_labels), activation="softmax", name="dips"
        )(F)

        # Input layer
        jet_inputs = Input(shape=(njet_features,))

        # Adding the intermediate dense layers for DL1
        x = jet_inputs
        for unit in NN_structure["intermediate_units"]:
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        # Concatenate the inputs
        x = Concatenate()([F, x])

        # Loop to initialise the hidden layers
        for unit in NN_structure["DL1_units"]:
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        jet_output = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
            name="umami",
        )(x)

        umami = Model(
            inputs=[trk_inputs, jet_inputs], outputs=[dips_output, jet_output]
        )

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    # Set optimier and loss
    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    umami.compile(
        loss="categorical_crossentropy",
        loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )

    return umami, NN_structure["epochs"]


def Dips_model(train_config=None, input_shape=None):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    if train_config.model_file is not None:
        # Load DIPS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        dips = load_model(train_config.model_file, {"Sum": Sum}, compile=False)

    else:
        logger.info("No modelfile provided! Initialize a new one!")

        # Set the track input
        trk_inputs = Input(shape=input_shape)

        # Masking the missing tracks
        masked_inputs = Masking(mask_value=0)(trk_inputs)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(NN_structure["ppm_sizes"]):

            tdd = TimeDistributed(
                Dense(phi_nodes, activation="linear"), name=f"Phi{i}_Dense"
            )(tdd)

            if batch_norm:
                tdd = TimeDistributed(
                    BatchNormalization(), name=f"Phi{i}_BatchNormalization"
                )(tdd)

            if dropout != 0:
                tdd = TimeDistributed(
                    Dropout(rate=dropout), name=f"Phi{i}_Dropout"
                )(tdd)

            tdd = TimeDistributed(
                Activation(activations.relu), name=f"Phi{i}_ReLU"
            )(tdd)

        # This is where the magic happens... sum up the track features!
        F = Sum(name="Sum")(tdd)

        # Define the main dips structure
        for j, (F_nodes, p) in enumerate(
            zip(
                NN_structure["dense_sizes"],
                [dropout] * len(NN_structure["dense_sizes"][:-1]) + [0],
            )
        ):

            F = Dense(F_nodes, activation="linear", name=f"F{j}_Dense")(F)
            if batch_norm:
                F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
            if dropout != 0:
                F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
            F = Activation(activations.relu, name=f"F{j}_ReLU")(F)

        # Set output and activation function
        output = Dense(
            len(class_labels), activation="softmax", name="Jet_class"
        )(F)
        dips = Model(inputs=trk_inputs, outputs=output)

    # Print Dips model summary when log level lower or equal INFO level
    if logger.level <= 20:
        dips.summary()

    # Set optimier and loss
    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    dips.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )
    return dips, NN_structure["epochs"]


def DL1_model(train_config, input_shape):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    # Load model from file if defined
    if train_config.model_file is not None:
        logger.info(f"Loading model from: {train_config.model_file}")
        model = load_model(train_config.model_file, compile=False)

    else:
        # Define input
        inputs = Input(shape=input_shape)

        # Define layers
        for i, unit in enumerate(NN_structure["dense_sizes"]):
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(inputs)

            # Add Batch Normalization if True
            if batch_norm:
                x = BatchNormalization()(x)

            # Add dropout if != 0
            if dropout != 0:
                x = Dropout(NN_structure["dropout_rate"][i])(x)

            # Define activation for the layer
            x = Activation(NN_structure["activations"][i])(x)

        predictions = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
        )(x)
        model = Model(inputs=inputs, outputs=predictions)

    # Print DL1 model summary when log level lower or equal INFO level
    if logger.level <= 20:
        model.summary()

    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )
    return model, NN_structure["epochs"]


def Deepsets_model(
    repeat_input_shape,
    num_conditions,
    num_set_features,
    sets_nodes,
    classif_nodes,
    classif_output,
    pooling: str = "Max",
    attention_nodes: list = None,
    condition_sets: bool = True,
    condition_attention: bool = True,
    condition_classifier: bool = True,
    shortcut_inputs: bool = True,
    sets_batch_norm: bool = False,
    classif_batch_norm: bool = False,
    activation: str = "relu",
    attention_softmax: bool = True,
):

    # Get the sets input
    repeat_inputs = Input(shape=repeat_input_shape)

    # Check for condition sets
    if condition_sets is True:
        # Get the condtion input
        condition_inputs = Input(shape=(num_conditions,))

        # Get conditional deep sets layer
        layers = ConditionalDeepSet(
            sets_nodes + [num_set_features],
            activation=activation,
            batch_norm=sets_batch_norm,
            name="CondDeepSets",
        )([repeat_inputs, condition_inputs])

    else:
        # Get normal Deep sets_layer
        layers = DeepSet(
            sets_nodes + [num_set_features],
            activation=activation,
            batch_norm=sets_batch_norm,
            name="DeepSets",
        )(repeat_inputs)

    # Check for shortcutting
    if shortcut_inputs:
        layers = Concatenate()([layers, repeat_inputs])

    # Get the correct pooling
    pooling = pooling.lower()

    # Ensure that on of these pooling operations is used
    assert pooling in [
        "max",
        "average",
        "mean",
        "sum",
        "attention",
    ], "Pooling must be either max, mean, average, sum, or attention."

    # Check for max pooling
    if pooling == "max":
        pooled = GlobalMaxPool1D()(layers)

    # Check for attention pooling
    elif pooling in "attention":

        # Assert that attention nodes are given and that the attention_nodes are
        # in list form
        assert (
            attention_nodes is not None and type(attention_nodes) is list
        ), "Please specify number of nodes for the attention layers in a list."

        # If condition attention, gtet conditional attention layers
        if condition_attention is True:
            attention = ConditionalAttention(
                attention_nodes,
                apply_softmax=attention_softmax,
                name="CondAttention",
            )([repeat_inputs, condition_inputs])

        # Else get "normal" attention layers
        else:
            attention = Attention(
                attention_nodes,
                apply_softmax=attention_softmax,
                name="Attention",
            )(repeat_inputs)

        # Pool the outputs with attention pooling
        pooled = AttentionPooling()([attention, layers])

    # Check for sum pooling
    elif pooling == "sum":
        pooled = Lambda(lambda x: K.sum(x, axis=1))(layers)

    # Else use masked average pooling
    else:
        pooled = MaskedAverage1DPooling()(layers)

    # If the conditional classifer is true, concatenate
    # the conditional inputs and the pooled outputs of the deep sets part
    if condition_classifier is True:
        pooled = Concatenate()([condition_inputs, pooled])

    # Get the dense net which further processes the output of the deep sets
    output = DenseNet(
        classif_nodes,
        classif_output,
        activation=activation,
        batch_norm=classif_batch_norm,
        name="DenseNet",
    )(pooled)

    # Check if conditional inputs are needed somewhere and build model
    if (
        condition_attention and (pooling == "attention")
    ) or condition_classifier is True:
        deepsets = Model(
            inputs=[repeat_inputs, condition_inputs], outputs=output
        )

    # Else build "normal" deep sets model
    else:
        deepsets = Model(inputs=repeat_inputs, outputs=output)

    # Return model
    return deepsets
