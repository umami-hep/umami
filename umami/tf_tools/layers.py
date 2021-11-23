"""
Implementations by Johnny Raine
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Dense, Layer


class DenseNet(Layer):
    """
    Define a DenseNet as a layer to easier access it.
    """

    def __init__(
        self,
        nodes,
        output_nodes=1,
        activation="relu",
        batch_norm=False,
        **kwargs,
    ):
        # Define the attributes
        self.nodes = nodes
        self.output_nodes = output_nodes
        self.activation = activation
        self.batch_norm = batch_norm

        # Define the layer structure
        self.layers = []

        # Iterate over given layers
        for node in nodes[:-1]:

            # Append dense layer with activation
            self.layers.append(Dense(units=node, activation=activation))

            # Apply batch normalisation if wanted
            if batch_norm is True:
                self.layers.append(BatchNormalization())

        # Add final dense layer unit
        self.layers.append(Dense(units=nodes[-1], activation=activation))

        # Define output activation function based on output node size
        output_activation = "sigmoid" if output_nodes == 1 else "softmax"
        self.layers.append(
            Dense(units=output_nodes, activation=output_activation)
        )

        # Assert that there are nodes defined
        assert len(nodes), "No layers in DenseNet"
        super().__init__(**kwargs)

    def call(self, inputs):
        out = self.layers[0](inputs)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

    def get_config(self):
        # Get configuration of the network
        config = {
            "nodes": self.nodes,
            "output_nodes": self.output_nodes,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
        }
        base_config = super(DenseNet, self).get_config()

        # Return a dict with the configurations
        return dict(list(base_config.items()) + list(config.items()))


class DeepSet(Layer):
    """
    Define a deep set layer for easier usage.
    """

    def __init__(
        self,
        nodes,
        activation="relu",
        batch_norm=False,
        mask_zero=True,
        **kwargs,
    ):
        # Define attributes
        self.nodes = nodes
        self.activation = activation
        self.batch_norm = batch_norm
        self.mask_zero = mask_zero
        self.supports_masking = True

        # Define the layer structure
        self.layers = []

        # Iterate over the given nodes
        for node in nodes[:-1]:

            # Append Dense node with activation
            self.layers.append(Dense(units=node, activation=activation))

            # Apply batch normalisation if active
            if batch_norm is True:
                self.layers.append(BatchNormalization())

        # Append final dense layer with activation
        self.layers.append(Dense(units=nodes[-1], activation=activation))

        # Check that nodes are in layers
        assert len(self.layers), "No layers in DeepSet"
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        # Assert that the tensor shape is at least rank 3
        assert (
            len(inputs.shape) == 3
        ), "DeepSets layer requires tensor of rank 3. Shape of tensor received {}".format(
            inputs.shape
        )

        # Check if mask is None and the standard zero mask is used
        if mask is None and self.mask_zero:

            # Compute zero mask
            mask = self.compute_mask(inputs, mask)

        # Define out
        out = self.layers[0](inputs)
        for layer in self.layers[1:]:
            out = layer(out)

        # if mask is not None:
        #    out *= (1-K.cast(mask,dtype="float32"))

        return out

    def compute_mask(self, inputs, mask=None):

        # Check if mask zero is true
        if not self.mask_zero:
            return None

        else:
            # Return correct masking
            return K.equal(K.sum(inputs ** 2, axis=-1), 0)

    def get_config(self):
        # Get configuration of the network
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "mask_zero": self.mask_zero,
        }
        base_config = super(DeepSet, self).get_config()

        # Return a dict with the configurations
        return dict(list(base_config.items()) + list(config.items()))


class MaskedSoftmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        # Get attributes
        self.axis = axis
        self.supports_masking = True
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        # Check for masking
        if mask is None:

            # Compute masking for not existing inputs
            mask = self.compute_mask(inputs)

        # Calculate Softmax
        inputs = K.exp(inputs) * (1 - K.cast(mask, dtype="float32"))

        # Return Masked Softmax
        return inputs / K.sum(inputs, axis=1, keepdims=True)

    def compute_mask(self, inputs, mask=None):
        # Return mask
        return K.equal(inputs, 0)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(MaskedSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):
    """
    Define Attention Layer.
    """

    def __init__(
        self,
        nodes,
        activation="relu",
        mask_zero=True,
        apply_softmax=True,
        **kwargs,
    ):
        self.nodes = nodes
        self.activation = activation
        self.mask_zero = mask_zero
        self.apply_softmax = apply_softmax

        self.layers = []
        for node in nodes:
            self.layers.append(Dense(units=node, activation=activation))
        self.layers.append(Dense(units=1, activation="sigmoid"))
        assert len(self.layers), "No layers in DeepSet"
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        assert (
            len(inputs.shape) == 3
        ), "Attention layer requires tensor of rank 3. Shape of tensor received {}".format(
            inputs.shape
        )

        if mask is None and self.mask_zero:
            mask = self.compute_mask(inputs, mask)

        attention = self.layers[0](inputs)
        for layer in self.layers[1:]:
            attention = layer(attention)

        attention = K.squeeze(attention, -1)

        if mask is not None:
            attention *= 1 - K.cast(mask, dtype="float32")

        if self.apply_softmax:
            attention_out = MaskedSoftmax(axis=1)(attention)
            # attention_out = Softmax(axis=1)(attention)
            return attention_out
        else:
            return attention

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        else:
            return K.equal(K.sum(inputs ** 2, axis=-1), 0)

    def get_config(self):
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "mask_zero": self.mask_zero,
            "apply_softmax": self.apply_softmax,
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionPooling(Layer):
    """
    Define Attention Pooling Layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):

        # Get attention and feature tensor
        attention, features = inputs[:2]

        # Assert correct shape
        assert (len(attention.shape) == 2) & (
            len(features.shape) == 3
        ), "Please provide attention tensor as first argument (rank 2), followed by feature tensor (rank 3)"

        # Pool with dot product
        pool = K.batch_dot(K.expand_dims(attention, 1), features)
        pool = K.squeeze(pool, -2)

        # Return pooled
        return pool


class ConditionalAttention(Layer):
    """
    Define custom conditional attention layer. This uses the standard attention
    with a condition to further improve the attention weighing.
    """

    def __init__(
        self,
        nodes,
        activation="relu",
        mask_zero=True,
        apply_softmax=True,
        **kwargs,
    ):
        # Define attributes
        self.nodes = nodes
        self.activation = activation
        self.mask_zero = mask_zero
        self.apply_softmax = apply_softmax

        # Get the basic attention layer
        self.attention = Attention(
            nodes, activation, mask_zero, apply_softmax, **kwargs
        )

        # Get the layers from attention
        self.layers = self.attention.layers
        super().__init__(**kwargs)

    def call(self, inputs):

        # Retrieve repeated vector and condition vector
        repeat, condition = inputs[:2]

        # Assert correct shapes
        assert (len(repeat.shape) == 3) & (
            len(condition.shape) == 2
        ), "Repeated vector must be rank 3 input, condition vector must be rank 2. Tensors provided have shapes {}, {}".format(
            repeat.shape, condition.shape
        )

        # Get the number of repeat
        nrepeat = repeat.shape[1]

        # Change the shape of the inputs to fit them together
        condition = K.tile(K.expand_dims(condition, -2), (1, nrepeat, 1))

        # Concatenate condition and repeat info
        inputs = K.concatenate([condition, repeat], axis=-1)

        # Compute mask if used
        mask = self.compute_mask(repeat) if self.mask_zero else None

        # Get the attention output
        attention_out = self.attention(inputs, mask=mask)

        # Return attention output
        return attention_out

    def compute_mask(self, inputs, mask=None):
        # Check for mask
        if not self.mask_zero:
            return None

        else:
            # Return mask if used
            return K.equal(K.sum(inputs ** 2, axis=-1), 0)

    def get_config(self):
        # Get the configs of the layer as dict
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "mask_zero": self.mask_zero,
            "apply_softmax": self.apply_softmax,
        }
        base_config = super(ConditionalAttention, self).get_config()

        # Return dict of the configs
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalDeepSet(Layer):
    def __init__(
        self,
        nodes,
        activation="relu",
        batch_norm=False,
        mask_zero=True,
        **kwargs,
    ):
        # Get attributes
        self.nodes = nodes
        self.activation = activation
        self.batch_norm = batch_norm
        self.mask_zero = mask_zero
        self.supports_masking = True

        # Get a DeepSet layer with the correct attributes
        self.deepsets = DeepSet(
            nodes, activation, batch_norm, mask_zero, **kwargs
        )

        # Get layers from deep sets layer
        self.layers = self.deepsets.layers
        super().__init__(**kwargs)

    def call(self, inputs):

        # Get repeated vector and conditions vector
        repeat, condition = inputs[:2]

        # Assert correct shape of the repeated and conditions vector
        assert (len(repeat.shape) == 3) & (
            len(condition.shape) == 2
        ), "Repeated vector must be rank 3 input, condition vector must be rank 2. Tensors provided have shapes {}, {}".format(
            repeat.shape, condition.shape
        )

        # Get the number of repeat vectors
        nrepeat = repeat.shape[1]

        # Extend the dimension of conditions so it fits with the repeated vector
        condition = K.tile(K.expand_dims(condition, -2), (1, nrepeat, 1))

        # Conacatenate condtions and repeated vectors
        inputs = K.concatenate([condition, repeat], axis=-1)

        # Calculate mask if needed
        mask = self.compute_mask(repeat) if self.mask_zero else None

        # Get deepsets output
        deepsets_out = self.deepsets(inputs, mask=mask)

        # Retrun conditional deep sets output
        return deepsets_out

    def compute_mask(self, inputs, mask=None):

        # Check if masking is zero
        if not self.mask_zero:
            return None

        else:
            # Return masking
            return K.equal(K.sum(inputs ** 2, axis=-1), 0)

    def get_config(self):
        # Get the configs of the layer as dict
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "mask_zero": self.mask_zero,
        }
        base_config = super(ConditionalDeepSet, self).get_config()

        # Return dict of the configs
        return dict(list(base_config.items()) + list(config.items()))


class MaskedAverage1DPooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        # Check for masking
        if mask is not None:

            # Create mask based on given mask
            mask1 = K.cast(mask, dtype="float32")

        else:
            # Create standard mask with zero as value
            mask1 = K.cast(K.equal(K.sum(inputs ** 2, axis=-1), 0), "float32")

        # Multiply inputs with new mask which is formed to correct dimension
        inputs = inputs * K.tile(
            K.expand_dims((1 - K.cast(mask, dtype="float32")), -1),
            [1, 1, K.shape(inputs)[-1]],
        )

        # Sum over inputs
        sumd = K.sum(inputs, axis=1)

        # Get number of tracks in correct shape
        ntrack = K.tile(
            K.expand_dims(K.sum(1 - mask1, axis=1), -1), [1, K.shape(sumd)[-1]]
        )

        # Return mean of sumd and number of track
        mean = sumd / ntrack
        return mean


class Sum(Layer):
    """
    Simple sum layer.
    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on their
    own.

    Author: Dan Guest
    https://github.com/dguest/flow-network/blob/master/SumLayer.py

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:, :, None]
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None
