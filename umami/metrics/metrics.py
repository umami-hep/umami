#!/usr/bin/env python

"""
Script with all the metrics calculations used to grade the
performance of the taggers.
"""

import copy

import numpy as np

# Try to import keras from tensorflow
try:
    import tensorflow.keras.backend as K

except ModuleNotFoundError:
    pass


def CalcDiscValues(
    jets_dict: dict,
    index_dict: dict,
    main_class: str,
    frac_dict: dict,
    rej_class: str = None,
) -> np.ndarray:
    """
    Calculate the discriminant values of the given jets for
    the given main class.

    Parameters
    ----------
    jets_dict : dict
        Dict with the jets inside. In each entry are the jets
        of one class and their output values of the NN as
        numpy ndarray in the shape (nJets, n_outputs).
    index_dict : dict
        Dict with the class names as keys and their corresponding
        column number in the n_outputs.
    main_class : str
        String of the main class. "bjets" for b-tagging.
    frac_dict : dict
        Dict with the fractions used to calculate the disc score.
        The values in here needs to add up to one!
    rej_class : str
        Name of the class of jets for which the discriminant values
        are to be computed.

    Returns
    -------
    disc_score : numpy.ndarray
        Array with the discriminant score values for the jets.

    Nodes
    -----
    The function calculates the discriminant values for the jets
    with the following equation:

    .. math::
        D_b = \\ln \\left(\\frac{p_b}{f_c * p_c + f_u * p_u} \\right)

    This is done here for the special case of 3 classes where bjets is
    the main class (signal class) and cjets and ujets are the background
    classes. The values :math:`f_c` and :math:`f_u` are taken from the
    frac_dict. The key is the class name, cjets for example, and the value
    is a float with the value of :math:`f_c`.

    Examples
    --------
    >>> jets_dict = {"bjets": np.array([[0.1, 0.1, 0.8], [0.0, 0.1, 0.9]]),
    ...              "cjets": np.array([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]]),
    ...              "ujets": np.array([[0.9, 0.1, 0.0], [0.7, 0.2, 0.1]])}
    {'bjets': array([[0.1, 0.1, 0.8],
        [0. , 0.1, 0.9]]),
     'cjets': array([[0.2, 0.6, 0.2],
        [0.1, 0.8, 0.1]]),
     'ujets': array([[0.9, 0.1, 0. ],
        [0.7, 0.2, 0.1]])}

    >>> index_dict = {"bjets": 2, "cjets": 1, "ujets": 0}
    {'bjets': 2, 'cjets': 1, 'ujets': 0}

    >>> main_class = "bjets"
    'bjets'

    >>> frac_dict = {"cjets": 0.018, "ujets": 0.982}
    {'cjets': 0.018, 'ujets': 0.982}

    The following will output the discriminant values for the two given bjets.
    Note that if no rej_class is given, the discriminant values for the main
    class jets are calculated.

    >>> disc_score = CalcDiscValues(jets_dict=jets_dict,
    ...                             index_dict=index_dict,
    ...                             main_class=main_class,
    ...                             frac_dict=frac_dict)
    [2.07944154, 6.21460804]

    Now, we can calculate the discriminant values for the cjets class.

    >>> disc_score = CalcDiscValues(jets_dict=jets_dict,
    ...                             index_dict=index_dict,
    ...                             main_class=main_class,
    ...                             frac_dict=frac_dict,
    ...                             rej_class"cjets")
    [-0.03536714, -0.11867153]
    """

    # Set the rejection class for rejection calculation
    if rej_class is None:
        rej_class = main_class

    # Init denominator of disc_score and add_small
    denominator = 0
    add_small = 1e-10

    # Get class_labels list without main class
    class_labels_wo_main = list(jets_dict.keys())
    class_labels_wo_main.remove(main_class)

    # Calculate counter of disc_score
    counter = jets_dict[rej_class][:, index_dict[main_class]] + add_small

    # Calculate denominator of disc_score
    for class_label in class_labels_wo_main:
        denominator += (
            frac_dict[class_label] * jets_dict[rej_class][:, index_dict[class_label]]
        )
    denominator += add_small

    # Calculate final disc_score and return it
    return np.log(counter / denominator)


def GetScore(
    y_pred: np.ndarray,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    use_keras_backend: bool = False,
) -> np.ndarray:
    """
    Similar to CalcDiscValues but uses directly the output of the
    NN (shape: (nJets, nClasses)) for calculation.

    Parameters
    ----------
    y_pred : numpy.ndarray
        The prediction output of the NN.
    class_labels : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets".
    frac_dict : dict
        A dict with the respective fractions for each class provided
        except main_class.
    use_keras_backend : bool
        Decide, if the values are calculated with the keras backend
        or numpy (Keras is needed for the saliency maps).

    Returns
    -------
    disc_score : numpy.ndarray
        Discriminant Score for the jets provided.

    Examples
    --------
    >>> y_pred = np.array([[0.1, 0.1, 0.8],
    ...                    [0.0, 0.1, 0.9],
    ...                    [0.2, 0.6, 0.2],
    ...                    [0.1, 0.8, 0.1]])
    array([[0.1, 0.1, 0.8],
           [0. , 0.1, 0.9],
           [0.2, 0.6, 0.2],
           [0.1, 0.8, 0.1]])

    >>> class_labels = ["ujets", "cjets", "bjets"]
    ['ujets', 'cjets', 'bjets']

    >>> main_class = "bjets"
    'bjets'

    >>> frac_dict = {"cjets": 0.018, "ujets": 0.982}
    {'cjets': 0.018, 'ujets': 0.982}

    Now we can call the function which will return the discriminant values
    for the given jets based on their given NN outputs (y_pred).

    >>> disc_scores = GetScore(y_pred=y_pred,
    ...                        class_labels=class_labels,
    ...                        main_class=main_class,
    ...                        frac_dict=frac_dict)
    [2.07944154, 6.21460804, -0.03536714, -0.11867153]
    """

    # Check if y_pred and class_labels has the needed similar shapes
    assert np.shape(y_pred)[1] == len(class_labels)

    # Init index dict
    index_dict = {}

    # Get Index of main class
    for class_label in class_labels:
        index_dict[f"{class_label}"] = class_labels.index(class_label)

    # Init denominator of disc_score and add_small
    denominator = 0
    add_small = 1e-10

    # Get class_labels list without main class
    class_labels_wo_main = copy.deepcopy(class_labels)
    class_labels_wo_main.remove(main_class)

    # Calculate counter of disc_score
    counter = y_pred[:, index_dict[main_class]] + add_small

    # Calculate denominator of disc_score
    for class_label in class_labels_wo_main:
        denominator += frac_dict[class_label] * y_pred[:, index_dict[class_label]]
    denominator += add_small

    # Calculate final disc_score and return it
    if use_keras_backend is True:
        disc_value = K.log(counter / denominator)

    else:
        disc_value = np.log(counter / denominator)

    return disc_value


def discriminant_output_shape(input_shape: tuple) -> tuple:
    """
    Ensure the correct output shape of the discriminant.

    Parameters
    ----------
    input_shape : tuple
        Input shape that is used.

    Returns
    -------
    shape : tuple
        The shape of the first dimension of the input as tuple.
    """
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    return (shape[0],)


def get_gradients(
    model: object,
    X: np.ndarray,
    nJets: int,
):
    """
    Calculating the gradients with respect to the input variables.
    Note that only Keras backend functions can be used here because
    the gradients are tensorflow tensors and are not compatible with
    numpy.

    Parameters
    ----------
    model : object
        Loaded keras model.
    X : numpy.ndarray
        Track inputs of the jets.
    nJets : int
        Number of jets to be used.

    Returns
    -------
    gradients : tensorflow.Tensor
        Gradients of the network for the given inputs.
    """

    gradients = K.gradients(model.output, model.inputs)

    input_tensors = model.inputs + [K.learning_phase()]
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # Pass in the cts and categorical inputs, as well as the learning phase
    # (0 for test mode)
    gradients = compute_gradients([X[:nJets], 0])

    return gradients[0]


def GetRejection(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    target_eff: float,
):
    """
    Calculates the rejections for a specific WP for all provided jets
    with all classes except the discriminant class (main_class).
    You can't calculate the rejection for the signal class.

    Parameters
    ----------
    y_pred : numpy.ndarray
        The prediction output of the NN. This must be the shape of
        (nJets, nClasses).
    y_true : numpy.ndarray
        The true class of the jets. This must also be of the shape
        (nJets, nClasses) (One-Hot-encoded).
    class_labels : list
        A list of the class_labels which are used. This must be the
        same order as the truth! See the nodes for more details.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets".
    frac_dict : dict
        A dict with the respective fractions for each class provided
        except main_class.
    target_eff : float
        WP which is used for discriminant calculation.

    Returns
    -------
    Rejection_Dict : dict
        Dict of the rejections. The keys of the dict
        are the provided class_labels without main_class
    cut_value : float
        Cut value that is calculated for the given working point.

    Raises
    ------
    ZeroDivisionError
        If no jets which passes the cut value are given. E.g. if
        no light jet is passing the WP cut, the rejection would
        be infinite.
    ValueError
        If the given y_true does not match the provided class_labels.
    ValueError
        If the given shape of y_true is not supported!

    Nodes
    -----
    The function calculates the discriminant values for the given jets
    with the following equation:

    .. math::
        D_b = \\ln \\left(\\frac{p_b}{f_c * p_c + f_u * p_u} \\right)

    This is done here for the special case of 3 classes where bjets is
    the main class (signal class) and cjets and ujets are the background
    classes. The values :math:`f_c` and :math:`f_u` are taken from the
    frac_dict. The key is the class name, cjets for example, and the value
    is a float with the value of :math:`f_c`.

    The class_labels MUST be the same order as the one hot encoded truth.
    So when [0, 0, 1] is the y_true for one jet and the first column is for
    the ujets, the second for the cjets and the third for the bjets, then the
    class_labels list MUST be ["ujets", "cjets", "bjets"].

    Examples
    --------
    >>> y_pred = np.array([[0.1, 0.1, 0.8],
    ...                    [0.0, 0.1, 0.9],
    ...                    [0.2, 0.6, 0.2],
    ...                    [0.1, 0.8, 0.1]])
    array([[0.1, 0.1, 0.8],
           [0. , 0.1, 0.9],
           [0.2, 0.6, 0.2],
           [0.1, 0.8, 0.1]])

    >>> y_true = np.array([[0, 0, 1],
    ...                    [0, 0, 1],
    ...                    [0, 1, 0],
    ...                    [0, 1, 0]])
    array([[0, 0, 1],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 0]])

    >>> class_labels = ["ujets", "cjets", "bjets"]
    ['ujets', 'cjets', 'bjets']

    >>> main_class = "bjets"
    'bjets'

    >>> frac_dict = {"cjets": 0.018, "ujets": 0.982}
    {'cjets': 0.018, 'ujets': 0.982}

    >>> target_beff = 0.30
    0.30

    The following will output the rejection for the given jets
    based on their NN outputs.

    >>> Rej_Dict = GetRejection(y_pred=y_pred,
    ...                         y_true=y_true,
    ...                         class_labels=class_labels,
    ...                         main_class=main_class,
    ...                         frac_dict=frac_dict,
    ...                         target_eff=target_eff)
    """

    # Assert that y_pred and y_true have the same shape
    if y_pred.shape == y_true.shape and (len(y_pred.shape) == 2):
        if not y_true.shape[1] == len(class_labels):
            raise ValueError(
                "The given y_true shape does not match the class labels! y_true shape:"
                f" {y_true.shape} \n, class_labels shape: {len(class_labels)}"
            )

        else:
            y_true = np.argmax(y_true, axis=1)

    elif len(y_true.shape) != 1:
        raise ValueError(f"Provided y_true has an unsupported shape {y_true.shape}")

    # Init new dict for jets and indices
    jets_dict = {}
    index_dict = {}
    rej_dict = {}

    # Iterate over the different class_labels and select their respective jets
    for class_counter, class_label in enumerate(class_labels):
        jets_dict.update({f"{class_label}": y_pred[y_true == class_counter]})
        index_dict.update({f"{class_label}": class_counter})

    # Calculate disc score
    disc_scores = CalcDiscValues(
        jets_dict=jets_dict,
        index_dict=index_dict,
        main_class=main_class,
        frac_dict=frac_dict,
        rej_class=None,
    )

    # Calculate cutvalue on the discriminant depending of the WP
    cutvalue = np.percentile(disc_scores, 100.0 * (1.0 - target_eff))

    # Get all non-main flavours
    class_labels_wo_main = copy.deepcopy(class_labels)
    class_labels_wo_main.remove(main_class)

    # Calculate efficiencies
    for iter_main_class in class_labels_wo_main:
        try:
            rej_dict[f"{iter_main_class}_rej"] = 1 / (
                len(
                    jets_dict[iter_main_class][
                        CalcDiscValues(
                            jets_dict=jets_dict,
                            index_dict=index_dict,
                            main_class=main_class,
                            frac_dict=frac_dict,
                            rej_class=iter_main_class,
                        )
                        > cutvalue
                    ]
                )
                / (len(jets_dict[iter_main_class]) + 1e-10)
            )

        except ZeroDivisionError as Error:
            raise ZeroDivisionError(
                "Not enough jets for rejection calculation of class "
                f"{iter_main_class} for {target_eff} efficiency!\n"
                "Maybe loosen the eff_min to fix it or give more jets!"
            ) from Error

    return rej_dict, cutvalue
