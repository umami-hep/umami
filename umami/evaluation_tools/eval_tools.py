#!/usr/bin/env python

"""
Script with all the higher level evaluation functions.
"""

from umami.configuration import global_config, logger  # isort:skip
import copy

import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

import umami.metrics as umt


def GetRejectionPerEfficiencyDict(
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

    Returns
    -------
    tagger_rej_dicts : dict
        Dict with the rejections for each tagger/class (wo main),
        disc cuts per effs and the effs.
    """

    # Get flavour categories
    flavour_categories = global_config.flavour_categories

    logger.info("Calculating rejections per efficiency")
    effs = np.linspace(eff_min, eff_max, x_axis_granularity)

    # Prepare lists of class_labels without main and tagger with freshly trained
    class_labels_wo_main = copy.deepcopy(class_labels)
    class_labels_wo_main.remove(main_class)
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
    for eff in effs:
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

            rej_dict_tmp, disc_cut_dict_tmp = umt.GetRejection(
                y_pred=y_pred,
                y_true=y_true,
                class_labels=class_labels,
                main_class=main_class,
                frac_dict=frac_values[f"{tagger}"]
                if tagger in tagger_names
                else frac_values_comp[f"{tagger}"],
                target_eff=eff,
            )

            tagger_disc_cut_dicts[f"disc_{tagger}"].append(disc_cut_dict_tmp)
            for rej_type, _ in rej_dict_tmp.items():
                tagger_rej_dicts[f"{tagger}_{rej_type}"].append(rej_dict_tmp[rej_type])

    # Remove double entries and print warning
    skipped_taggers = list(dict.fromkeys(skipped_taggers))
    logger.warning(f"Following taggers are skipped for file: {skipped_taggers}")

    # Remove entries of not loaded taggers from the dicts
    for skipped_tagger in skipped_taggers:
        del tagger_disc_cut_dicts[f"disc_{skipped_tagger}"]

        for rej in class_labels_wo_main:
            del tagger_rej_dicts[f"{skipped_tagger}_{rej}_rej"]

    return {**tagger_disc_cut_dicts, **tagger_rej_dicts}


def GetScoresProbsDict(
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

    # Adding trained tagger probabilities
    for counter, tagger in enumerate(tagger_names + tagger_list):

        # Make a copy of the class_labels
        class_labels_copy = copy.deepcopy(class_labels)

        for flav_index, flav in enumerate(class_labels):
            if tagger in tagger_names:
                df_discs_dict[
                    f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                ] = tagger_preds[tagger_names.index(tagger)][:, flav_index]

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
                    class_labels_copy.remove(flav)

            # Check if tagger is in file
            if len(class_labels_copy) == 0:
                logger.warning(f"Tagger {tagger} not in .h5 files! Skipping...")
                continue

            # Reshape to wrong sorted (transpose change it to correct shape)
            y_pred = tmp.reshape((len(class_labels_copy), -1))
            y_pred = np.transpose(y_pred)

        # Adding scores of the trained network
        try:
            df_discs_dict[f"disc_{tagger}"] = umt.GetScore(
                y_pred=y_pred,
                class_labels=class_labels_copy,
                main_class=main_class,
                frac_dict=frac_values[f"{tagger}"]
                if tagger in tagger_names
                else frac_values_comp[f"{tagger}"],
            )

        except KeyError:
            logger.warning(f"{tagger} is in files, but not in frac_dict! Skipping...")
            continue

    return df_discs_dict


def GetSaliencyMapDict(
    model: object,
    model_pred,
    X_test,
    Y_test,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    nJets: int = int(10e4),
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
    nJets : int
        Number of jets to use to calculate the saliency maps.

    Returns
    -------
    Map_dict : dict
        Dict with the saliency values
    """

    logger.info("Calculate gradients for inputs")

    # Cut off last layer of the model for saliency maps
    cutted_model = model.layers[-1].output

    # Define the last node for the discriminant output
    disc = Lambda(
        umt.GetScore,
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
    Disc_values = umt.GetScore(
        y_pred=model_pred,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
    )

    # Init small dict
    map_dict = {}

    # Get spartial class id
    class_indices = list(range(len(class_labels)))

    # Iterate over different beff, jet flavours and passed options
    for target_beff in [60, 70, 77, 85]:
        for (jet_flavour, class_index) in zip(class_labels, class_indices):
            for PassBool in [True, False]:

                # Get the Disc_values value for a specific flavour
                Disc_values_flavour = Disc_values[Y_test[:, class_index].astype(bool)]

                # Get the cutvalue for the specific WP
                cutvalue = np.percentile(Disc_values_flavour, (100 - target_beff))

                # Set PassBool masking
                if PassBool is True:
                    mask = Y_test[:, class_index].astype(bool)
                    mask = mask & (nTrks == 8)
                    mask = mask & (Disc_values > cutvalue)

                elif PassBool is False:
                    mask = Y_test[:, class_index].astype(bool)
                    mask = mask & (nTrks == 8)
                    mask = mask & (Disc_values < cutvalue)

                # Get gradient map
                gradient_map = umt.get_gradients(model, X_test[mask], nJets)

                # Turn gradient map for plotting
                gradient_map = np.swapaxes(gradient_map, 1, 2)

                # Mean over the jets
                gradient_map = np.mean(gradient_map, axis=0)

                map_dict.update(
                    {f"{target_beff}_{jet_flavour}_{PassBool}": gradient_map}
                )

    return map_dict


def RecomputeScore(
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
    return umt.GetScore(
        shaped_proba,
        class_labels=model_class_labels,
        main_class=main_class,
        frac_dict=model_frac_values,
    )
