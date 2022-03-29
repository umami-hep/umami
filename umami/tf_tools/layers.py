"""
Implementations by Johnny Raine
"""
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    BatchNormalization,
    Dense,
    Layer,
)


class DenseNet(Layer):
    """
    Define a DenseNet as a layer to easier access it.
    """

    def __init__(
        self,
        nodes: list,
        output_nodes: int = 1,
        activation: str = "relu",
        batch_norm: bool = False,
        **kwargs,
    ):
        """
        Init the DenseNet layer

        Parameters
        ----------
        nodes : list
            List with the number of neurons per node
        output_nodes : int
            Number of outputs in the output node
        activation : str, optional
            Activation which is used, by default "relu"
        batch_norm : bool, optional
            Use batch normalisation, by default False
        **kwargs : dict
            Additional arguments passed.
        """

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
        self.layers.append(Dense(units=output_nodes, activation=output_activation))

        # Assert that there are nodes defined
        assert len(nodes), "No layers in DenseNet"
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Define what happens when the layer is called

        Parameters
        ----------
        inputs : object
            Input to the network.

        Returns
        -------
        output : object
            Output of the network.
        """
        out = self.layers[0](inputs)
        for layer in self.layers[1 : len(self.layers) - 1]:
            out = layer(out)
        out_last = self.layers[-1](out)
        return out, out_last

    def get_config(self) -> dict:
        """
        Return the settings of the network.

        Returns
        -------
        dict
            Dict with the config settings.
        """
        # Get configuration of the network
        config = {
            "nodes": self.nodes,
            "output_nodes": self.output_nodes,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
        }
        base_config = super().get_config()

        # Return a dict with the configurations
        return dict(list(base_config.items()) + list(config.items()))


class DeepSet(Layer):
    """
    Define a deep set layer for easier usage.
    """

    def __init__(
        self,
        nodes: list,
        activation: str = "relu",
        batch_norm: bool = False,
        mask_zero: bool = True,
        **kwargs,
    ):
        """
        Init the DeepSet Layer.

        Parameters
        ----------
        nodes : list
            List with the number of neurons per node
        activation : str, optional
            Activation which is used, by default "relu"
        batch_norm : bool, optional
            Use batch normalisation, by default False
        mask_zero : bool, optional
            Use 0 as mask value, by default True
        **kwargs : dict
            Additional arguments passed.
        """
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
        assert self.layers, "No layers in DeepSet"
        super().__init__(**kwargs)

    def call(self, inputs, mask: float = None):  # pylint: disable=arguments-differ
        """
        Return the output of the network for a given input.

        Parameters
        ----------
        inputs : object
            Input to layer.
        mask : float, optional
            Mask value, by default None

        Returns
        -------
        output
            Layer output.
        """
        # Assert that the tensor shape is at least rank 3
        assert len(inputs.shape) == 3, (
            "DeepSets layer requires tensor of rank 3. Shape of tensor"
            f" received {inputs.shape}"
        )

        # Check if mask is None and the standard zero mask is used
        if mask is None and self.mask_zero:

            # Compute zero mask
            mask = self.compute_mask(inputs, mask)

        # Define out
        out = self.layers[0](inputs)
        for layer in self.layers[1:]:
            out = layer(out)

        return out

    def compute_mask(
        self, inputs, mask: float = None
    ):  # pylint: disable=unused-argument
        """
        Compute the masking.

        Parameters
        ----------
        inputs : object
            Input to a layer.
        mask : float
            Custom mask value (needed in tensorflow).

        Returns
        -------
        masking
            Return correct masking
        """

        # Check if mask zero is true
        if not self.mask_zero:
            return None

        # Return correct masking
        return K.equal(K.sum(inputs**2, axis=-1), 0)

    def get_config(self):
        """
        Return the settings of the network.

        Returns
        -------
        dict
            Dict with the config settings.
        """
        # Get configuration of the network
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "mask_zero": self.mask_zero,
        }
        base_config = super().get_config()

        # Return a dict with the configurations
        return dict(list(base_config.items()) + list(config.items()))


class MaskedSoftmax(Layer):
    """Softmax layer with masking."""

    def __init__(self, axis=-1, **kwargs):
        """
        Init masked softmax layer

        Parameters
        ----------
        axis : int, optional
            Which axis is used for softmax, by default -1
        **kwargs : dict
            Additional arguments passed.
        """
        # Get attributes
        self.axis = axis
        self.supports_masking = True
        super().__init__(**kwargs)

    def call(self, inputs, mask: float = None):  # pylint: disable=arguments-differ
        """
        Return the output of the softmax layer.

        Parameters
        ----------
        inputs : object
            Layer input.
        mask : float, optional
            Masking value, by default None

        Returns
        -------
        output
            Return output of the layer.
        """
        # Check for masking
        if mask is None:

            # Compute masking for not existing inputs
            mask = self.compute_mask(inputs, mask)

        # Calculate Softmax
        inputs = K.exp(inputs) * (1 - K.cast(mask, dtype="float32"))

        # Return Masked Softmax
        return inputs / K.sum(inputs, axis=1, keepdims=True)

    def compute_mask(
        self, inputs, mask: float = None
    ):  # pylint: disable=no-self-use,unused-argument
        """
        Compute mask.

        Parameters
        ----------
        inputs : object
            Layer input.
        mask : float
            Custom mask value (needed in tensorflow).

        Returns
        -------
        masking
            Masking for the given input.
        """
        # Return mask
        return K.equal(inputs, 0)

    def get_config(self):
        """
        Return the settings of the network.

        Returns
        -------
        dict
            Dict with the config settings.
        """
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):
    """
    Define Attention Layer.
    """

    def __init__(
        self,
        nodes: list,
        activation: str = "relu",
        mask_zero: bool = True,
        apply_softmax: bool = True,
        **kwargs,
    ):
        """
        Init the Attention layer

        Parameters
        ----------
        nodes : list
            List with the number of neurons per node
        activation : str, optional
            Activation which is used, by default "relu"
        mask_zero : bool, optional
            Use 0 as mask value, by default True
        apply_softmax : bool, optional
            Use softmax, by default True
        **kwargs : dict
            Additional arguments passed.
        """
        self.nodes = nodes
        self.activation = activation
        self.mask_zero = mask_zero
        self.apply_softmax = apply_softmax

        self.layers = []
        for node in nodes:
            self.layers.append(Dense(units=node, activation=activation))
        self.layers.append(Dense(units=1, activation="sigmoid"))
        assert self.layers, "No layers in DeepSet"
        super().__init__(**kwargs)

    def call(self, inputs, mask: float = None):  # pylint: disable=arguments-differ
        """
        Return the output of the network for a given input.

        Parameters
        ----------
        inputs : object
            Input to layer.
        mask : float, optional
            Mask value, by default None

        Returns
        -------
        output
            Layer output.
        """

        assert len(inputs.shape) == 3, (
            "Attention layer requires tensor of rank 3. Shape of tensor"
            f" received {inputs.shape}"
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
        return attention

    def compute_mask(
        self, inputs, mask: float = None
    ):  # pylint: disable=unused-argument
        """
        Compute mask.

        Parameters
        ----------
        inputs : object
            Layer input.
        mask : float
            Custom mask value (needed in tensorflow).

        Returns
        -------
        masking
            Masking for the given input.
        """
        if not self.mask_zero:
            return None

        return K.equal(K.sum(inputs**2, axis=-1), 0)

    def get_config(self):
        """
        Return the settings of the network.

        Returns
        -------
        dict
            Dict with the config settings.
        """
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "mask_zero": self.mask_zero,
            "apply_softmax": self.apply_softmax,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionPooling(Layer):  # pylint: disable=too-few-public-methods
    """
    Define Attention Pooling Layer.
    """

    def __init__(self, **kwargs):  # pylint: disable=useless-super-delegation
        """Init Attention Pooling layer

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed.
        """
        super().__init__(**kwargs)

    def call(self, inputs):  # pylint: disable=arguments-differ,no-self-use
        """
        Return the output of the layer for a given input.

        Parameters
        ----------
        inputs : object
            Input to layer.

        Returns
        -------
        output
            Layer output.
        """

        # Get attention and feature tensor
        attention, features = inputs[:2]

        # Assert correct shape
        assert (len(attention.shape) == 2) & (len(features.shape) == 3), (
            "Please provide attention tensor as first argument (rank 2),"
            " followed by feature tensor (rank 3)"
        )

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
        nodes: list,
        activation: str = "relu",
        mask_zero: bool = True,
        apply_softmax: bool = True,
        **kwargs,
    ):
        """
        Init the Conditional Attention Layer.

        Parameters
        ----------
        nodes : list
            List with the number of neurons per node
        activation : str, optional
            Activation which is used, by default "relu"
        mask_zero : bool, optional
            Use 0 as mask value, by default True
        apply_softmax : bool, optional
            Use softmax, by default True
        **kwargs : dict
            Additional arguments passed.
        """
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

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Return the output of the network for a given input.

        Parameters
        ----------
        inputs : object
            Input to layer.

        Returns
        -------
        output
            Layer output.
        """

        # Retrieve repeated vector and condition vector
        repeat, condition = inputs[:2]

        # Assert correct shapes
        assert (len(repeat.shape) == 3) & (len(condition.shape) == 2), (
            "Repeated vector must be rank 3 input, condition vector must be"
            f" rank 2. Tensors provided have shapes {repeat.shape},"
            f" {condition.shape}"
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

    def compute_mask(
        self, inputs, mask: float = None
    ):  # pylint: disable=unused-argument
        """
        Compute the masking.

        Parameters
        ----------
        inputs : object
            Input to a layer.
        mask : float
            Custom mask value (needed in tensorflow).

        Returns
        -------
        masking
            Return correct masking
        """

        # Check for mask
        if not self.mask_zero:
            return None

        # Return mask if used
        return K.equal(K.sum(inputs**2, axis=-1), 0)

    def get_config(self):
        """
        Return the settings of the network.

        Returns
        -------
        dict
            Dict with the config settings.
        """
        # Get the configs of the layer as dict
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "mask_zero": self.mask_zero,
            "apply_softmax": self.apply_softmax,
        }
        base_config = super().get_config()

        # Return dict of the configs
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalDeepSet(Layer):
    """Keras layer for conditional deep set."""

    def __init__(
        self,
        nodes: list,
        activation: str = "relu",
        batch_norm: bool = False,
        mask_zero: bool = True,
        **kwargs,
    ):
        """
        Init the DeepSet Layer.

        Parameters
        ----------
        nodes : list
            List with the number of neurons per node
        activation : str, optional
            Activation which is used, by default "relu"
        batch_norm : bool, optional
            Use batch normalisation, by default False
        mask_zero : bool, optional
            Use 0 as mask value, by default True
        **kwargs : dict
            Additional arguments passed.
        """

        # Get attributes
        self.nodes = nodes
        self.activation = activation
        self.batch_norm = batch_norm
        self.mask_zero = mask_zero
        self.supports_masking = True

        # Get a DeepSet layer with the correct attributes
        self.deepsets = DeepSet(nodes, activation, batch_norm, mask_zero, **kwargs)

        # Get layers from deep sets layer
        self.layers = self.deepsets.layers
        super().__init__(**kwargs)

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Return the output of the layer for a given input.

        Parameters
        ----------
        inputs : object
            Input to layer.

        Returns
        -------
        output
            Layer output.
        """

        # Get repeated vector and conditions vector
        repeat, condition = inputs[:2]

        # Assert correct shape of the repeated and conditions vector
        assert (len(repeat.shape) == 3) & (len(condition.shape) == 2), (
            "Repeated vector must be rank 3 input, condition vector must be"
            f" rank 2. Tensors provided have shapes {repeat.shape},"
            f" {condition.shape}"
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

    def compute_mask(
        self, inputs, mask: float = None
    ):  # pylint: disable=unused-argument
        """
        Compute the masking.

        Parameters
        ----------
        inputs : object
            Input to a layer.
        mask : float
            Custom mask value (needed in tensorflow).

        Returns
        -------
        masking
            Return correct masking
        """

        # Check if masking is zero
        if not self.mask_zero:
            return None

        # Return masking
        return K.equal(K.sum(inputs**2, axis=-1), 0)

    def get_config(self):
        """
        Return the settings of the network.

        Returns
        -------
        dict
            Dict with the config settings.
        """
        # Get the configs of the layer as dict
        config = {
            "nodes": self.nodes,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "mask_zero": self.mask_zero,
        }
        base_config = super().get_config()

        # Return dict of the configs
        return dict(list(base_config.items()) + list(config.items()))


class MaskedAverage1DPooling(Layer):  # pylint: disable=too-few-public-methods
    """Keras layer for masked 1D average pooling."""

    def __init__(self, **kwargs):  # pylint: disable=useless-super-delegation
        """Init the masked average 1d pooling layer.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed.
        """
        super().__init__(**kwargs)

    def call(
        self, inputs, mask: float = None
    ):  # pylint: disable=arguments-differ,no-self-use
        """
        Return the output of the layer for a given input.

        Parameters
        ----------
        inputs : object
            Input to layer.
        mask : float, optional
            Mask value, by default None

        Returns
        -------
        output
            Layer output.
        """
        # Check for masking
        if mask is not None:

            # Create mask based on given mask
            mask1 = K.cast(mask, dtype="float32")

        else:
            # Create standard mask with zero as value
            mask1 = K.cast(K.equal(K.sum(inputs**2, axis=-1), 0), "float32")

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
        """
        Init the class.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed.
        """
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):  # pylint: disable=unused-argument
        """Build step which is skipped.

        Parameters
        ----------
        input_shape : object
            Input shape of the layer (is needed in tensorflow).

        """
        pass  # pylint: disable=unnecessary-pass

    def call(self, x, mask: float = None):  # pylint: disable=no-self-use
        """
        Return the output of the layer.

        Parameters
        ----------
        x : object
            Layer input
        mask : float, optional
            Mask value, by default None

        Returns
        -------
        output
            Output of the layer
        """
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:, :, None]
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):  # pylint: disable=no-self-use
        """
        Compute the output shape.

        Parameters
        ----------
        input_shape : object
            Layer input shape

        Returns
        -------
        output
            Layer output.
        """
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):  # pylint: disable=no-self-use,unused-argument
        """Compute masking

        Parameters
        ----------
        inputs : object
            Layer input.
        mask : float
            Custom mask value (needed in tensorflow).

        Returns
        -------
        masking
            Return the masking
        """
        return None
