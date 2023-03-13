"""Script with all the higher level evaluation functions."""
from umami.configuration import global_config, logger  # isort:skip
import copy
from itertools import permutations

import numpy as np
from tensorflow.keras.layers import Lambda  # pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tqdm import tqdm

import umami.metrics as umt
from umami.preprocessing_tools import get_variable_dict
from umami.tools import check_main_class_input


def calculate_fraction_dict(
    class_labels_wo_main: list,
    frac_min: float,
    frac_max: float,
    step: float,
) -> list:
    """Return all combinations of fractions for the given background classes
    which adds up to one.

    Parameters
    ----------
    class_labels_wo_main : list
        List of the background classes.
    frac_min : float
        Minimum value of the fractions.
    frac_max : float
        Minimum value of the fractions.
    step : float
        Step size of the loop.

    Returns
    -------
    list
        List with the different dicts inside.

    Raises
    ------
    ValueError
        If no combination of fractions yields a sum of 1.
    """
    # Create a list for the dicts
    dict_list = []

    # Create the permutations
    combinations = permutations(
        np.arange(frac_min, frac_max + step, step),
        len(class_labels_wo_main),
    )

    # Iterate over the combinations
    for iterator in combinations:

        # Add up the values
        summation = np.sum(np.round(iterator, decimals=4))

        # Check if the values add up to one
        if summation == 1:
            # Round the values in the tuple to 4 decimals
            iterator = tuple(
                map(lambda x: isinstance(x, float) and round(x, 4) or x, iterator)
            )

            # Create a dict for the values
            tmp_dict = {}
            for counter, bkg_class in enumerate(class_labels_wo_main):
                tmp_dict[bkg_class] = iterator[counter]

            # Append the dict to list
            dict_list.append(tmp_dict)

    if len(dict_list) == 0:
        raise ValueError(
            "You defined a frac_min, frac_max, step wrong. No combination of the"
            " fractions produce them sum of 1. Please change frac_min, frac_max or"
            " step!"
        )

    # Return the complete list
    return dict_list


def get_rej_per_frac_dict(
    jets,
    y_true: np.ndarray,
    tagger_classes: list,
    tagger_preds: list,
    tagger_names: list,
    tagger_list: list,
    class_labels: list,
    main_class: str,
    target_eff: float,
    step: float = 0.01,
    frac_min: float = 0.0,
    frac_max: float = 1.0,
    progress_bar: bool = False,
) -> dict:
    """Calculate the rejections for the background classes for all possible
    combinations of fraction values of the background classes. The fractions
    need to add up to one to be valid.

    Parameters
    ----------
    jets : pandas.DataFrame
        Dataframe with jets and the probabilites of the comparison taggers as columns.
    y_true : numpy.ndarray
        Truth labels of the jets.
    tagger_classes: list
        List of the classes that were used to train the freshly trained tagger.
        For example, if you want to test the behavior of tau jets in the tagger
        although the tagger was not trained on taus.
    tagger_preds : list
        Prediction output of the taggers listed. [pred_dips, pred_umami]
    tagger_names : list
        Names of the freshly trained taggers. ["dips", "umami"]
    tagger_list : list
        List of the comparison tagger names.
    class_labels : list
        List of class labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    target_eff : float
        Target efficiency for which the rejections are calculated.
    step : float, optional
        Step size of the change of the fraction values, by default 0.01
    frac_min : float
        Minimum value of the fractions, by default 0.0.
    frac_max : float
        Minimum value of the fractions, by default 1.0.
    progress_bar : bool, optional
        Decide, if a progress bar for the different combinations is printed to
        the terminal. By default False.

    Returns
    -------
    dict
        Dict with the rejections for the taggers for the given fraction combinations.
    """
    # Check if a freshly trained tagger is given. If not, init empty lists so the
    # loop is not broken
    if tagger_names is None and tagger_preds is None:
        tagger_names = []
        tagger_preds = []

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get flavour categories
    flavour_categories = global_config.flavour_categories

    logger.info("Calculating rejections per fractions")

    # Get a deep copy of the class labels
    class_labels_wo_main = copy.deepcopy(list(dict.fromkeys(class_labels)))

    # Remove the main classes from the copy
    for m_class in main_class:
        class_labels_wo_main.remove(m_class)

    # Create the extended tagger list with fresh taggers and taggers from file
    extended_tagger_list = tagger_list + tagger_names

    # Init a dict where the results can be added to
    tagger_rej_dict = {}

    # Get the list with the combinations of the fractions as dicts
    dict_list = calculate_fraction_dict(
        class_labels_wo_main=class_labels_wo_main,
        frac_min=frac_min,
        frac_max=frac_max,
        step=step,
    )

    # Init a tagger-skipped dict
    skipped_taggers = {tagger: [] for tagger in extended_tagger_list}

    # Loop over effs for ROC plots
    for frac_dict in tqdm(dict_list, disable=not progress_bar):

        # Create dict entry key for the given fraction dict
        dict_key = ""

        # Loop over the items in the dict and make a string out of it
        for bkg_class, value in frac_dict.items():
            dict_key += f"{bkg_class}_{value}"

            if bkg_class != list(frac_dict.keys())[-1]:
                dict_key += "_"

        # Loop over the taggers
        for tagger in extended_tagger_list:

            # If the tagger is fresh, load the provided y_pred
            if tagger in tagger_names:
                y_pred = tagger_preds[tagger_names.index(tagger)]

                # Calculate the dimensions that need to be added
                extra_dim = len(set(class_labels) - set(tagger_classes))
                if extra_dim > 0:
                    skipped_taggers[tagger] += list(
                        set(class_labels) - set(tagger_classes)
                    )
                    y_pred = np.append(
                        y_pred,
                        np.zeros(shape=(extra_dim, y_pred.shape[0])).transpose(),
                        axis=1,
                    )

            # If the tagger is from the files, load the probabilities
            else:
                # Shape the probabilities of the comparison taggers like the output of
                # the networks
                for flav_index, flav in enumerate(class_labels):

                    # Trying to load the output probs of the tagger from file
                    try:
                        # Append the output to a flat array
                        if flav_index == 0:
                            tmp = jets[
                                f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                            ].values

                        else:
                            tmp = np.append(
                                tmp,
                                jets[
                                    f"{tagger}_"
                                    f'{flavour_categories[flav]["prob_var_name"]}'
                                ].values,
                            )

                    except KeyError:
                        skipped_taggers[tagger].append(flav)
                        if flav_index == 0:
                            tmp = np.zeros_like(y_true)

                        else:
                            tmp = np.append(tmp, np.zeros_like(y_true))

                # Reshape to wrong sorted (transpose change it to correct shape)
                y_pred = tmp.reshape((len(class_labels), -1))
                y_pred = np.transpose(y_pred)

            # Calculate the rejections for the given tagger
            rej_dict_tmp, _ = umt.get_rejection(
                y_pred=y_pred,
                y_true=y_true,
                class_labels=class_labels,
                main_class=main_class,
                frac_dict=frac_dict,
                target_eff=target_eff,
            )

            # Store the rejections and the fraction values in a new dict
            tagger_rej_dict[f"{tagger}_{dict_key}"] = {**rej_dict_tmp, **frac_dict}

    # Check which flavours are not present for which tagger
    tagger_to_remove = []
    masked_taggers = {}
    for iter_tagger, labels in skipped_taggers.items():
        if set(labels) == set(class_labels):
            tagger_to_remove.append(iter_tagger)

        elif len(labels) != 0:
            masked_taggers[iter_tagger] = list(set(labels))

    if len(masked_taggers.keys()) != 0:
        logger.warning(
            "The following taggers have at least one class not in their output. This "
            "missing output is masked with zeros:"
        )
        for iter_tagger, labels in masked_taggers.items():
            logger.warning("Tagger: %s, Missing ouputs: %s", iter_tagger, labels)

    if len(tagger_to_remove) != 0:
        logger.warning(
            "Following taggers are not present in the h5 files and are skipped: %s",
            tagger_to_remove,
        )

    # Remove taggers that are not present in files
    tagger_rej_dict_keys = list(tagger_rej_dict.keys())
    for remove_tagger in tagger_to_remove:
        for tagger_dict_key in tagger_rej_dict_keys:
            if tagger_dict_key.startwith(remove_tagger):
                tagger_rej_dict.pop(tagger_dict_key)

    return tagger_rej_dict


def get_rej_per_eff_dict(
    jets,
    y_true,
    tagger_classes: list,
    tagger_preds: list,
    tagger_names: list,
    tagger_list: list,
    class_labels: list,
    main_class: str,
    frac_values: dict,
    frac_values_comp: dict,
    x_axis_granularity: int = 100,
    eff_min: float = 0.49,
    eff_max: float = 1.0,
    progress_bar: bool = False,
) -> dict:
    """
    Calculates the rejections for the classes and taggers provided for
    different efficiencies of the main class.

    Parameters
    ----------
    jets : pandas.DataFrame
        Dataframe with jets and the probabilites of the comparison taggers as columns.
    y_true : numpy.ndarray
        Truth labels of the jets.
    tagger_classes: list
        List of the classes that were used to train the freshly trained tagger.
        For example, if you want to test the behavior of tau jets in the tagger
        although the tagger was not trained on taus.
    tagger_preds : list
        Prediction output of the taggers listed. [pred_dips, pred_umami]
    tagger_names : list
        Names of the freshly trained taggers. ["dips", "umami"]
    tagger_list : list
        List of the comparison tagger names.
    class_labels : list
        List of class labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    frac_values : dict
        Dict with the fraction values for the fresh taggers.
    frac_values_comp : dict
        Dict with the fraction values for the comparison taggers.
    x_axis_granularity : int
        Granularity of the efficiencies.
    eff_min : float
        Lowest value for the efficiencies linspace.
    eff_max : float
        Highst value for the efficiencies linspace.
    progress_bar : bool, optional
        Decide, if a progress bar for the different effs is printed to
        the terminal. By default False.

    Returns
    -------
    tagger_rej_dicts : dict
        Dict with the rejections for each tagger/class (wo main),
        disc cuts per effs and the effs.
    """
    # Check if a freshly trained tagger is given. If not, init empty lists so the
    # loop is not broken
    if tagger_names is None and tagger_preds is None:
        tagger_names = []
        tagger_preds = []

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get flavour categories
    flavour_categories = global_config.flavour_categories

    logger.info("Calculating rejections per efficiency")
    effs = np.linspace(eff_min, eff_max, x_axis_granularity)

    # Get a deep copy of the class labels
    class_labels_wo_main = copy.deepcopy(list(dict.fromkeys(class_labels)))

    # Remove the main classes from the copy
    for m_class in main_class:
        class_labels_wo_main.remove(m_class)

    # Create the extended tagger list with fresh taggers and taggers from file
    extended_tagger_list = tagger_list + tagger_names

    tagger_rej_dicts = {
        f"{tagger}_{rej}_rej": []
        for rej in class_labels_wo_main
        for tagger in extended_tagger_list
    }
    tagger_rej_dicts["effs"] = effs
    tagger_disc_cut_dicts = {f"disc_{tagger}": [] for tagger in extended_tagger_list}

    # Init a tagger-skipped dict
    skipped_taggers = {tagger: [] for tagger in extended_tagger_list}

    # Loop over effs for ROC plots
    for eff in tqdm(effs, disable=not progress_bar):

        # Loop over the taggers which are to be evaluated
        for tagger in extended_tagger_list:

            # Check if the tagger is a freshly trained tagger
            if tagger in tagger_names:
                tagger_pred_tmp = tagger_preds[tagger_names.index(tagger)]

                # Iterate over the flavours which are to be evaluated
                for iter_counter, iter_class in enumerate(class_labels):

                    # Try to get the probability output values for the given class
                    try:
                        if iter_counter == 0:
                            y_pred = tagger_pred_tmp[
                                :, tagger_classes.index(iter_class)
                            ]

                        else:
                            y_pred = np.append(
                                y_pred,
                                tagger_pred_tmp[:, tagger_classes.index(iter_class)],
                            )

                    # If no probabilty output values are available for the given class,
                    # mask them with zeros
                    except ValueError:
                        skipped_taggers[tagger].append(iter_class)
                        if iter_counter == 0:
                            y_pred = np.zeros(tagger_pred_tmp.shape[0])

                        else:
                            y_pred = np.append(
                                y_pred,
                                np.zeros(tagger_pred_tmp.shape[0]),
                            )

                # Reshape and transpose
                y_pred = y_pred.reshape((len(class_labels), -1))
                y_pred = np.transpose(y_pred)

            # Check if the tagger is a comparison tagger (from h5 file)
            else:
                # Shape the probabilities of the comparison taggers like the output of
                # the networks
                for flav_index, flav in enumerate(class_labels):

                    # Trying to load the output probs of the tagger from file
                    try:
                        # Append the output to a flat array
                        if flav_index == 0:
                            tmp = jets[
                                f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                            ].values

                        else:
                            tmp = np.append(
                                tmp,
                                jets[
                                    f"{tagger}_"
                                    f'{flavour_categories[flav]["prob_var_name"]}'
                                ].values,
                            )

                    except KeyError:
                        skipped_taggers[tagger].append(flav)
                        if flav_index == 0:
                            tmp = np.zeros_like(y_true)

                        else:
                            tmp = np.append(tmp, np.zeros_like(y_true))

                # Reshape to wrong sorted (transpose change it to correct shape)
                y_pred = tmp.reshape((len(class_labels), -1))
                y_pred = np.transpose(y_pred)

            rej_dict_tmp, disc_cut_dict_tmp = umt.get_rejection(
                y_pred=y_pred,
                y_true=y_true,
                class_labels=class_labels,
                main_class=main_class,
                frac_dict=(
                    frac_values[f"{tagger}"]
                    if tagger in tagger_names
                    else frac_values_comp[f"{tagger}"]
                ),
                target_eff=eff,
            )

            tagger_disc_cut_dicts[f"disc_{tagger}"].append(disc_cut_dict_tmp)
            for rej_type, _ in rej_dict_tmp.items():
                tagger_rej_dicts[f"{tagger}_{rej_type}"].append(
                    rej_dict_tmp[  # pylint: disable=unnecessary-dict-index-lookup
                        rej_type
                    ]
                )

    # Check which flavours are not present for which tagger
    tagger_to_remove = []
    masked_taggers = {}
    for iter_tagger, labels in skipped_taggers.items():
        if set(labels) == set(class_labels):
            tagger_to_remove.append(iter_tagger)

        elif len(labels) != 0:
            masked_taggers[iter_tagger] = list(set(labels))

    if len(masked_taggers.keys()) != 0:
        logger.warning(
            "The following taggers have at least one class not in their output. This "
            "missing output is masked with zeros:"
        )
        for iter_tagger, labels in masked_taggers.items():
            logger.warning("Tagger: %s, Missing ouputs: %s", iter_tagger, labels)

    if len(tagger_to_remove) != 0:
        logger.warning(
            "Following taggers are not present in the h5 files and are skipped: %s",
            tagger_to_remove,
        )

    # Remove taggers that are not present in files
    for iter_tagger in tagger_to_remove:
        del tagger_disc_cut_dicts[f"disc_{iter_tagger}"]

        for rej in class_labels_wo_main:
            del tagger_rej_dicts[f"{iter_tagger}_{rej}_rej"]

    return {**tagger_disc_cut_dicts, **tagger_rej_dicts}


def get_scores_probs_dict(
    jets,
    y_true,
    tagger_classes: list,
    tagger_preds: list,
    tagger_names: list,
    tagger_list: list,
    class_labels: list,
    main_class: str,
    frac_values: dict,
    frac_values_comp: dict,
) -> dict:
    """
    Get the probabilites in a new dict and calculate the discriminant scores.

    Parameters
    ----------
    jets : pandas.DataFrame
        Dataframe with the probabilites of the comparison taggers as columns
    y_true : numpy.ndarray
        Internal truth labeling of the used jets.
    tagger_classes: list
        List of the classes that were used to train the freshly trained tagger.
        For example, if you want to test the behavior of tau jets in the tagger
        although the tagger was not trained on taus.
    tagger_preds : list
        Prediction output of the taggers listed. e.g. [pred_dips, pred_umami]
    tagger_names : list
        Names of the freshly trained taggers. e.g. ["dips", "umami"]
    tagger_list : list
        List of the comparison tagger names.
    class_labels : list
        List of class labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    frac_values : dict
        Dict with the fraction values for the fresh taggers.
    frac_values_comp : dict
        Dict with the fraction values for the comparison taggers.

    Returns
    -------
    df_discs_dict : dict
        Dict with the discriminant scores of each jet and the probabilities
        of the different taggers for the used jets.
    """

    logger.info("Calculate discriminant scores")

    # Get flavour categories
    flavour_categories = global_config.flavour_categories

    # Define new dict with the evaluation results
    df_discs_dict = {
        "pt": jets[global_config.pTvariable],
        "eta": jets[global_config.etavariable],
        "labels": y_true,
    }

    # Check if a freshly trained tagger is given. If not, init empty lists so the
    # loop is not broken
    if tagger_names is None and tagger_preds is None:
        tagger_names = []
        tagger_preds = []

    # Create the extended tagger list with fresh taggers and taggers from file
    extended_tagger_list = tagger_list + tagger_names

    # Init a tagger-skipped dict
    skipped_taggers = {tagger: [] for tagger in extended_tagger_list}

    # Adding trained tagger probabilities
    for tagger in extended_tagger_list:

        # Get probability values of freshly trained tagger from provided predictions
        for flav_index, flav in enumerate(class_labels):
            if tagger in tagger_names:
                try:
                    df_discs_dict[
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                    ] = tagger_preds[tagger_names.index(tagger)][:, flav_index]

                except IndexError:
                    df_discs_dict[
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                    ] = np.zeros_like(y_true)

            else:
                # Get probablility values of comparison taggers from dataframe
                # (from file)
                try:
                    df_discs_dict[
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                    ] = jets[f'{tagger}_{flavour_categories[flav]["prob_var_name"]}']

                except KeyError:
                    df_discs_dict[
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                    ] = np.zeros_like(y_true)

        # Check which tagger is used
        if tagger in tagger_names:

            # Define y_pred for the freshly trained tagger
            y_pred = tagger_preds[tagger_names.index(tagger)]

            # Calculate the dimensions that need to be added
            extra_dim = len(set(class_labels) - set(tagger_classes))
            if extra_dim > 0:
                skipped_taggers[tagger] += list(set(class_labels) - set(tagger_classes))
                y_pred = np.append(
                    y_pred,
                    np.zeros(shape=(extra_dim, y_pred.shape[0])).transpose(),
                    axis=1,
                )

        else:
            # Shape the probabilities of the comparison taggers like the output of
            # the networks
            for flav_index, flav in enumerate(class_labels):

                # Trying to load the output probs of the tagger from file
                try:
                    # Append the output to a flat array
                    if flav_index == 0:
                        tmp = jets[
                            f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                        ].values

                    else:
                        tmp = np.append(
                            tmp,
                            jets[
                                f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                            ].values,
                        )

                except KeyError:
                    skipped_taggers[tagger].append(flav)
                    if flav_index == 0:
                        tmp = np.zeros_like(y_true)

                    else:
                        tmp = np.append(tmp, np.zeros_like(y_true))

            # Reshape to wrong sorted (transpose change it to correct shape)
            y_pred = tmp.reshape((len(class_labels), -1))
            y_pred = np.transpose(y_pred)

        # Adding scores of the trained network
        try:
            df_discs_dict[f"disc_{tagger}"] = umt.get_score(
                y_pred=y_pred,
                class_labels=class_labels,
                main_class=main_class,
                frac_dict=(
                    frac_values[f"{tagger}"]
                    if tagger in tagger_names
                    else frac_values_comp[f"{tagger}"]
                ),
            )

        except KeyError:
            logger.warning("%s is in files, but not in frac_dict! Skipping...", tagger)
            continue

    # Remove double entries and print warning
    tagger_to_remove = []
    masked_taggers = {}
    for iter_tagger, labels in skipped_taggers.items():
        if set(labels) == set(class_labels):
            tagger_to_remove.append(iter_tagger)

        elif len(labels) != 0:
            masked_taggers[iter_tagger] = list(set(labels))

    if len(masked_taggers.keys()) != 0:
        logger.warning(
            "The following taggers have at least one class not in their output. This "
            "missing output is masked with zeros:"
        )
        for iter_tagger, labels in masked_taggers.items():
            logger.warning("Tagger: %s, Missing ouputs: %s", iter_tagger, labels)

    if len(tagger_to_remove) != 0:
        logger.warning(
            "Following taggers are not present in the h5 files and are skipped: %s",
            tagger_to_remove,
        )

    return df_discs_dict


def get_saliency_map_dict(
    model: object,
    model_pred: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    var_dict_path: str,
    tracks_name: str,
    n_trks: int = None,
    effs: list = None,
    n_jets: int = int(10e4),
) -> dict:
    """
    Calculating the saliency maps dict.

    Parameters
    ----------
    model : object
        Loaded Keras model.
    model_pred : numpy.ndarray
        Model predictions of the model.
    x_test : numpy.ndarray
        Inputs to the model.
    y_test : numpy.ndarray
        Truth labels in one-hot-encoded format.
    class_labels : list
        List of class labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets".
    frac_dict : dict
        Dict with the fraction values for the tagger.
    var_dict_path : str
        Path to the variable dict which was used for training the tagger
        (to retrieve the inputs).
    tracks_name : str
        Name of the tracks which are used in the training.
    n_trks : int
        Number of tracks each jet needs to have. Saliency maps can
        only be calculated for a fixed number of tracks per jet.
        Only jets with this amount of tracks are used for calculation.
    effs : list, optional
        List with the efficiencies which are tested.
        If None is given, the default WPs of 60, 70, 77 and 85
        are tested. By default None.
    n_jets : int, optional
        Number of jets to use to calculate the saliency maps.
        By default 10e4

    Returns
    -------
    Map_dict : dict
        Dict with the saliency values

    Raises
    ------
    ValueError
        If given efficiencies are neither a list nor a int.
    """

    logger.info("Calculate gradients for inputs")

    # Check effs for None
    if effs is None:
        effs = [60, 70, 77, 85]

    elif isinstance(effs, int):
        effs = [effs]

    elif not isinstance(effs, list):
        raise ValueError(
            "Efficiencies for saliency calculation must be a list "
            f"or an int! Given type: {type(effs)}"
        )

    # Cut off last layer of the model for saliency maps
    cutted_model = model.layers[-1].output

    # Define the last node for the discriminant output
    disc = Lambda(
        umt.get_score,
        output_shape=umt.discriminant_output_shape,
        arguments={
            "class_labels": class_labels,
            "main_class": main_class,
            "frac_dict": frac_dict,
            "use_keras_backend": True,
        },
    )(cutted_model)

    # Define the computation graph for the model
    model = Model(model.inputs, disc)

    # Define boolean mask to filter placeholder tracks
    bool_mask = (np.sum(x_test, axis=-1) != 0).astype(bool)

    # Define the number of true tracks per jet as a mask
    n_trks_true = np.sum(bool_mask, axis=-1)

    # Get score for the dips prediction
    disc_values = umt.get_score(
        y_pred=model_pred,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
    )

    # Load the variable dict
    var_dict = get_variable_dict(var_dict_path)

    # Extract track variables from the dict
    trk_variables_dict = var_dict["track_train_variables"][tracks_name]

    # Flatten the track variables in one list
    trk_variables_list = [i for j in trk_variables_dict for i in trk_variables_dict[j]]

    # Init small dict
    map_dict = {"Variables_list": trk_variables_list}

    # Get spartial class id
    class_indices = list(range(len(class_labels)))

    # Iterate over different beff, jet flavours and passed options
    for target_beff in effs:
        for (jet_flavour, class_index) in zip(class_labels, class_indices):
            for pass_bool in [True, False]:

                # Get the Disc_values value for a specific flavour
                disc_values_flavour = disc_values[y_test[:, class_index].astype(bool)]

                # Get the cutvalue for the specific WP
                cutvalue = np.percentile(disc_values_flavour, (100 - target_beff))

                # Check for correct flavour and number of tracks
                mask = y_test[:, class_index].astype(bool)
                mask = mask & (n_trks == n_trks_true)

                # Set PassBool masking
                if pass_bool:
                    mask = mask & (disc_values > cutvalue)

                else:
                    mask = mask & (disc_values < cutvalue)

                # Get gradient map
                gradient_map = umt.get_gradients(model, x_test[mask], n_jets)

                # Turn gradient map for plotting
                gradient_map = np.swapaxes(gradient_map, 1, 2)

                # Mean over the jets
                gradient_map = np.mean(gradient_map, axis=0)

                map_dict.update(
                    {f"{target_beff}_{jet_flavour}_{pass_bool}": gradient_map}
                )

    return map_dict


def recompute_score(
    df_probs,
    model_tagger: str,
    main_class: str,
    model_frac_values: dict,
    model_class_labels: list,
):
    """
    Recompute the output scores of a given tagger.

    Parameters
    ----------
    df_probs : pandas.DataFrame
        Dataframe with the tagger probabilities inside.
    model_tagger : str
        Name of the tagger to use.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets".
    model_frac_values : dict
        Dict with the fraction values for the given model.
    model_class_labels : list
        List with the class labels which are to be used.

    Returns
    -------
    Scores : numpy.ndarray
        Array with the tagger scores for the given jets.
    """

    # Get the flavour categories
    flavour_categories = global_config.flavour_categories

    # Shape the probability dataframe
    for flav_index, flav in enumerate(model_class_labels):
        if flav_index == 0:
            unshaped_proba = df_probs[
                f"{model_tagger}_{flavour_categories[flav]['prob_var_name']}"
            ]
        else:
            unshaped_proba = np.append(
                unshaped_proba,
                df_probs[
                    f"{model_tagger}_{flavour_categories[flav]['prob_var_name']}"
                ].values,
            )
    # Reshape to wrong sorted (transpose change it to correct shape)
    shaped_proba = unshaped_proba.reshape((len(model_class_labels), -1))
    shaped_proba = np.transpose(shaped_proba)

    # Returns the score
    return umt.get_score(
        shaped_proba,
        class_labels=model_class_labels,
        main_class=main_class,
        frac_dict=model_frac_values,
    )
