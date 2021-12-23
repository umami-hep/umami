"""
Implementations by Johnny Raine
"""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, GlobalMaxPool1D, Input, Lambda
from tensorflow.keras.models import Model

from .layers import (
    Attention,
    AttentionPooling,
    ConditionalAttention,
    ConditionalDeepSet,
    DeepSet,
    DenseNet,
    MaskedAverage1DPooling,
)


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
        deepsets = Model(inputs=[repeat_inputs, condition_inputs], outputs=output)

    # Else build "normal" deep sets model
    else:
        deepsets = Model(inputs=repeat_inputs, outputs=output)

    # Return model
    return deepsets
