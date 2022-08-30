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
        Sum = np.sum(np.round(iterator, decimals=4))

        # Check if the values add up to one
        if Sum == 1:
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
    tagger_preds: list,
    tagger_names: list,
    tagger_list: list,
    class_labels: list,
    main_class: str,
    target_eff: float,
    step: float = 0.01,
    frac_min: float = 0.01,
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
        Minimum value of the fractions, by default 0.01.
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

    # Init a tagger-skipped list
    skipped_taggers = []

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

            # If the tagger is from the files, load the probabilities
            else:
                try:
                    y_pred = jets[
                        [
                            f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                            for flav in class_labels
                        ]
                    ].values

                except KeyError:
                    # Skipping this tagger if not in all flavours
                    # or the tagger present in file
                    logger.debug(tagger)
                    skipped_taggers.append(tagger)
                    continue

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

    # Remove double entries and print warning
    skipped_taggers = list(set(skipped_taggers))

    if skipped_taggers:
        logger.warning(
            "Taggers which do not have probability values for all requested class "
            "labels are not evaluated."
        )

    # Check if taggers where skipped and print them
    if len(skipped_taggers) != 0:
        logger.warning("Following taggers are skipped for file: %s", skipped_taggers)

    return tagger_rej_dict


def get_rej_per_eff_dict(
    jets,
    y_true,
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

    # Init a tagger-skipped list
    skipped_taggers = []

    # Loop over effs for ROC plots
    for eff in tqdm(effs, disable=not progress_bar):
        for tagger in extended_tagger_list:
            if tagger in tagger_names:
                y_pred = tagger_preds[tagger_names.index(tagger)]

            else:
                try:
                    y_pred = jets[
                        [
                            f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                            for flav in class_labels
                        ]
                    ].values

                except KeyError:
                    # Skipping this tagger if not in all flavours
                    # or the tagger present in file
                    skipped_taggers.append(tagger)
                    continue

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

    # Remove double entries and print warning
    skipped_taggers = list(dict.fromkeys(skipped_taggers))
    if skipped_taggers:
        logger.warning(
            "Taggers which do not have probability values for all requested class "
            "labels are not evaluated."
        )

    if len(skipped_taggers) != 0:
        logger.warning("Following taggers are skipped for file: %s", skipped_taggers)

    # Remove entries of not loaded taggers from the dicts
    for skipped_tagger in skipped_taggers:
        del tagger_disc_cut_dicts[f"disc_{skipped_tagger}"]

        for rej in class_labels_wo_main:
            del tagger_rej_dicts[f"{skipped_tagger}_{rej}_rej"]

    return {**tagger_disc_cut_dicts, **tagger_rej_dicts}


def get_scores_probs_dict(
    jets,
    y_true,
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

    # Adding trained tagger probabilities
    for counter, tagger in enumerate(tagger_names + tagger_list):

        # Make a copy of the class_labels
        class_labels_copy = copy.deepcopy(class_labels)

        for flav_index, flav in enumerate(class_labels):
            # Get probability values of freshly trained tagger from provided predictions
            if tagger in tagger_names:
                df_discs_dict[
                    f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                ] = tagger_preds[tagger_names.index(tagger)][:, flav_index]

            # Get probablility values of comparison taggers from dataframe (from file)
            else:
                try:
                    df_discs_dict[
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                    ] = jets[f'{tagger}_{flavour_categories[flav]["prob_var_name"]}']

                except KeyError:
                    # Skipping not available taggers probabilities
                    continue

        if tagger in tagger_names:
            y_pred = tagger_preds[counter]

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
                    logger.warning(
                        "Did not find probability values of flavour %s "
                        "for tagger %s. This is ignored.",
                        flav,
                        tagger,
                    )
                    class_labels_copy.remove(flav)

            # Check if tagger is in file
            if len(class_labels_copy) == 0:
                logger.warning("Tagger %s not in .h5 files! Skipping...", tagger)
                continue

            # Reshape to wrong sorted (transpose change it to correct shape)
            y_pred = tmp.reshape((len(class_labels_copy), -1))
            y_pred = np.transpose(y_pred)

        # Adding scores of the trained network
        try:
            df_discs_dict[f"disc_{tagger}"] = umt.get_score(
                y_pred=y_pred,
                class_labels=class_labels_copy,
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

    return df_discs_dict


def get_saliency_map_dict(
    model: object,
    model_pred: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    var_dict_path: str,
    tracks_name: str,
    nTracks: int = None,
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
    X_test : numpy.ndarray
        Inputs to the model.
    Y_test : numpy.ndarray
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
    nTracks : int
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

    # Check if default nTracks must be used
    if nTracks is None:
        nTracks = 8

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
    boolMask = (np.sum(X_test, axis=-1) != 0).astype(bool)

    # Define the number of true tracks per jet as a mask
    nTrks = np.sum(boolMask, axis=-1)

    # Get score for the dips prediction
    Disc_values = umt.get_score(
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
            for PassBool in [True, False]:

                # Get the Disc_values value for a specific flavour
                Disc_values_flavour = Disc_values[Y_test[:, class_index].astype(bool)]

                # Get the cutvalue for the specific WP
                cutvalue = np.percentile(Disc_values_flavour, (100 - target_beff))

                # Check for correct flavour and number of tracks
                mask = Y_test[:, class_index].astype(bool)
                mask = mask & (nTrks == nTracks)

                # Set PassBool masking
                if PassBool:
                    mask = mask & (Disc_values > cutvalue)

                else:
                    mask = mask & (Disc_values < cutvalue)

                # Get gradient map
                gradient_map = umt.get_gradients(model, X_test[mask], n_jets)

                # Turn gradient map for plotting
                gradient_map = np.swapaxes(gradient_map, 1, 2)

                # Mean over the jets
                gradient_map = np.mean(gradient_map, axis=0)

                map_dict.update(
                    {f"{target_beff}_{jet_flavour}_{PassBool}": gradient_map}
                )

    return map_dict


def recompute_score(
    df,
    model_tagger: str,
    main_class: str,
    model_frac_values: dict,
    model_class_labels: list,
):
    """
    Recompute the output scores of a given tagger.

    Parameters
    ----------
    df : pandas.DataFrame
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
            unshaped_proba = df[
                f"{model_tagger}_{flavour_categories[flav]['prob_var_name']}"
            ]
        else:
            unshaped_proba = np.append(
                unshaped_proba,
                df[
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
