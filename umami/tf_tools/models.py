"""
Implementations by Johnny Raine
"""

import os

import tensorflow.keras.backend as K  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    GlobalMaxPool1D,
    Input,
    Lambda,
)
from tensorflow.keras.models import Model, load_model  # pylint: disable=import-error

from umami.configuration import logger
from umami.tools import natural_keys

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


def prepare_model(
    train_config: object,
    continue_training: bool = False,
):
    """Prepare the keras model.

    Parameters
    ----------
    train_config : object
        Loaded train config file.
    continue_training : bool, optional
        Decide if the training is continued (True) or if the given model is used
        as start weights (False), by default False

    Returns
    -------
    model
        Loaded keras model (either the latest model for continuation or the given
        one).
    init_epoch
        Internal epoch number for the training. If the training is continued, this
        is the number of the lastest trained epoch (this is so that when the
        training starts, epoch 1 is not overwritten).
    load_optimiser
        Decide, if the optimiser of the model is loaded (True) or if the model
        will be recompiled.

    Raises
    ------
    ValueError
        If load_optimiser is True and no model file is given.
    """
    # Load NN Structure and training parameter from file
    load_optimiser = (
        train_config.NN_structure["load_optimiser"]
        if "load_optimiser" in train_config.NN_structure
        else False
    )

    # Check that load optimiser is only valid when a model file is given
    if load_optimiser is True and train_config.model_file is None:
        raise ValueError(
            "You can't load the optimiser state from a model if not model is given!"
        )

    # Init the init_epoch
    init_epoch = 0

    if train_config.model_file is not None:
        logger.info(f"Loading model from: {train_config.model_file}")
        model_file = train_config.model_file

    elif continue_training:
        # Get the lastest epoch available
        model_file_name = sorted(
            os.listdir(os.path.join(train_config.model_name, "model_files")),
            key=natural_keys,
        )[-1]

        # Load the latest model
        logger.info(f"Continue training using model {model_file_name}")

        # Set the load_optimiser to True so the model is not recompiled
        load_optimiser = True

        # Get the number of the last epoch which will be the init epoch
        init_epoch = int(
            model_file_name[
                model_file_name.rfind("model_epoch")
                + len("model_epoch") : model_file_name.rfind(".h5")
            ]
        )

        # Get the path of the selected model
        model_file = os.path.join(
            train_config.model_name,
            "model_files",
            model_file_name,
        )

    else:
        model_file = None

    # Check if the model file is found/given or not.
    if model_file:
        model = load_model(
            model_file,
            {
                "Sum": Sum,
                "Attention": Attention,
                "DeepSet": DeepSet,
                "AttentionPooling": AttentionPooling,
                "DenseNet": DenseNet,
                "ConditionalAttention": ConditionalAttention,
                "ConditionalDeepSet": ConditionalDeepSet,
            },
            compile=load_optimiser,
        )

    else:
        model = None

    # Return the model, init_epoch and the bool for loading of the
    # optimiser
    return model, init_epoch, load_optimiser


def Deepsets_model(
    repeat_input_shape,
    num_conditions: int,
    num_set_features: int,
    sets_nodes: list,
    classif_nodes: list,
    classif_output: int,
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
    """Keras implementation for deep sets

    Parameters
    ----------
    repeat_input_shape : tuple
        Input shape of the set data. For CADS, this
        would be the shape of the input to the deep
        sets part.
    num_conditions : int
        Number of conditions used for training.
    num_set_features : int
        Number of neurons of the last layer of the deep
        set.
    sets_nodes : list
        List with the number of neurons for the deep
        sets hidden layers.
    classif_nodes : list
        List with the number of neurons for the classification
        network (the F network after the deep sets part).
    classif_output : int
        Number of output neurons of the classification network
        (Number of classes used).
    pooling : str, optional
        poolig operation, by default "Max"
    attention_nodes : list, optional
        List of hidden layer neurons for the attention block,
        by default None.
    condition_sets : bool, optional
        Decide, if the conditional information is folded into
        the input of the deep sets block, by default True.
    condition_attention : bool, optional
        Decide, if the conditional information is folded into
        the input of the attention block, by default True
    condition_classifier : bool, optional.
        Decide, if the conditional information is folded into
        the input of the classification block, by default True
    shortcut_inputs : bool, optional
        Decide, if the deep sets inputs are short cutted (Adding
        it again after the deep sets block to the output of the
        deep sets block), by default True.
    sets_batch_norm : bool, optional
        Decide, if batch normalisation is used in the deep sets
        block, by default False.
    classif_batch_norm : bool, optional
        Decide, if batch normalisation is used in the classification
        block, by default False.
    activation : str, optional
        Decide, which activation function is used, by default "relu".
    attention_softmax : bool, optional
        Decide, if the output of the attention block is calculated
        using a sigmoid activation (False) or a Softmax activation
        (True), by default True.

    Returns
    -------
    keras model
        Model for deep sets training
    """

    # Get the sets input
    repeat_inputs = Input(shape=repeat_input_shape)

    # Get the condition inputs if needed
    if (
        condition_sets is True
        or condition_attention is True
        or condition_classifier is True
    ):
        # Get the condtion input
        condition_inputs = Input(shape=(num_conditions,))

    # Check for condition sets
    if condition_sets is True:
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
        assert attention_nodes is not None and isinstance(
            attention_nodes, list
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
        nodes=classif_nodes,
        output_nodes=classif_output,
        activation=activation,
        batch_norm=classif_batch_norm,
        class_output_only=True,
        name="DenseNet",
    )(pooled)

    # Check if conditional inputs are needed somewhere and build model
    if (
        (condition_attention and (pooling == "attention"))
        or condition_classifier
        or condition_sets
    ):
        deepsets = Model(inputs=[repeat_inputs, condition_inputs], outputs=output)

    # Else build "normal" deep sets model
    else:
        deepsets = Model(inputs=repeat_inputs, outputs=output)

    # Return model
    return deepsets


def Deepsets_model_umami(
    trk_input_shape,
    jet_input_shape,
    num_conditions,
    num_set_features,
    DIPS_sets_nodes,
    F_classif_nodes,
    classif_output,
    intermediate_units,
    DL1_units,
    pooling: str = "attention",
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
    """Keras implementation for umami with conditional attention

    Parameters
    ----------
    trk_input_shape : tuple
        Input shape of the set data. For UMAMI with conditional
        attention, this would be the shape of the input to the deep
        sets part.
    jet_input_shape : tuple
        Input shape of the jet data.
    num_conditions : int
        Number of conditions used for training.
    num_set_features : int
        Number of neurons of the last layer of the deep
        set.
    DIPS_sets_nodes : list
        List with the number of neurons for the deep
        sets hidden layers.
    F_classif_nodes : list
        List with the number of neurons for the classification
        network (the F network after the deep sets part).
    classif_output : int
        Number of output neurons of the classification network
        (Number of classes used).
    intermediate_units : list
        List with the number of neurons for the intermediate
        layer used for jet features
    DL1_units : list
        List with the number of neurons for the DL1r
        hidden layers.
    pooling : str, optional
        poolig operation, by default "Max"
    attention_nodes : list, optional
        List of hidden layer neurons for the attention block,
        by default None.
    condition_sets : bool, optional
        Decide, if the conditional information is folded into
        the input of the deep sets block, by default True.
    condition_attention : bool, optional
        Decide, if the conditional information is folded into
        the input of the attention block, by default True
    condition_classifier : bool, optional.
        Decide, if the conditional information is folded into
        the input of the classification block, by default True
    shortcut_inputs : bool, optional
        Decide, if the deep sets inputs are short cutted (Adding
        it again after the deep sets block to the output of the
        deep sets block), by default True.
    sets_batch_norm : bool, optional
        Decide, if batch normalisation is used in the deep sets
        block, by default False.
    classif_batch_norm : bool, optional
        Decide, if batch normalisation is used in the classification
        block, by default False.
    activation : str, optional
        Decide, which activation function is used, by default "relu".
    attention_softmax : bool, optional
        Decide, if the output of the attention block is calculated
        using a sigmoid activation (False) or a Softmax activation
        (True), by default True.

    Returns
    -------
    keras model
        Model for umami with conditional attention training
    """

    # Get the sets input
    trk_inputs = Input(shape=trk_input_shape)

    # Get the condition inputs if needed
    if (
        condition_sets is True
        or condition_attention is True
        or condition_classifier is True
    ):
        # Get the condtion input
        condition_inputs = Input(shape=(num_conditions,))

    # Check for condition sets
    if condition_sets is True:
        # Get conditional deep sets layer
        layers = ConditionalDeepSet(
            DIPS_sets_nodes + [num_set_features],
            activation=activation,
            batch_norm=sets_batch_norm,
            name="CondDeepSets",
        )([trk_inputs, condition_inputs])

    else:
        # Get normal Deep sets_layer
        layers = DeepSet(
            DIPS_sets_nodes + [num_set_features],
            activation=activation,
            batch_norm=sets_batch_norm,
            name="DeepSets",
        )(trk_inputs)

    # Check for shortcutting
    if shortcut_inputs:
        layers = Concatenate()([layers, trk_inputs])

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
        assert attention_nodes is not None and isinstance(
            attention_nodes, list
        ), "Please specify number of nodes for the attention layers in a list."

        # If condition attention, gtet conditional attention layers
        if condition_attention is True:
            attention = ConditionalAttention(
                attention_nodes,
                apply_softmax=attention_softmax,
                name="CondAttention",
            )([trk_inputs, condition_inputs])

        # Else get "normal" attention layers
        else:
            attention = Attention(
                attention_nodes,
                apply_softmax=attention_softmax,
                name="Attention",
            )(trk_inputs)

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
    sec_to_last, dips_output = DenseNet(
        F_classif_nodes,
        classif_output,
        activation=activation,
        batch_norm=classif_batch_norm,
        class_output_only=False,
        name="dips",
    )(pooled)

    # Check if conditional inputs are needed somewhere and build model
    jets_inputs = Input(shape=jet_input_shape)

    x = jets_inputs
    for unit in intermediate_units:
        x = Dense(
            units=unit,
            activation="linear",
            kernel_initializer="glorot_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    umami_in = Concatenate()([sec_to_last, x])

    umami_out = DenseNet(
        DL1_units,
        classif_output,
        activation=activation,
        batch_norm=classif_batch_norm,
        class_output_only=True,
        name="umami",
    )(umami_in)

    if (
        (condition_attention and (pooling == "attention"))
        or condition_classifier
        or condition_sets
    ):
        deepsets = Model(
            inputs=[trk_inputs, condition_inputs, jets_inputs],
            outputs=[dips_output, umami_out],
        )

    # Else build "normal" deep sets model
    else:
        deepsets = Model(
            inputs=[trk_inputs, jets_inputs], outputs=[dips_output, umami_out]
        )

    return deepsets
