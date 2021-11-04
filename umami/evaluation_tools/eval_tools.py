from umami.configuration import global_config, logger  # isort:skip
import copy

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

import umami.train_tools as utt


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
):
    """
    Calculates the rejections for the classes and taggers provided for
    different efficiencies of the main class.

    Input:
    - jets: Dataframe with the probabilites of the comparison taggers as columns
    - tagger_preds: Prediction output of the taggers listed. [pred_dips, pred_umami]
    - tagger_names: Names of the freshly trained taggers. ["dips", "umami"]
    - tagger_list: List of the comparison tagger names.
    - class_labels: List of class labels which are used.
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - frac_values: Dict with the fraction values for the fresh taggers.
    - frac_values_comp: Dict with the fraction values for the comparison taggers.
    - x_axis_granularity: Granularity of the efficiencies.
    - eff_min: Lowest value for the efficiencies linspace.
    - eff_max: Highst value for the efficiencies linspace.

    Output:
    - tagger_rej_dicts: Dict with the rejections for each tagger/class (wo main),
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
    tagger_disc_cut_dicts = {
        f"disc_{tagger}": [] for tagger in extended_tagger_list
    }

    # Loop over effs for ROC plots
    for eff in effs:
        for tagger in extended_tagger_list:
            if tagger in tagger_names:
                y_pred = tagger_preds[tagger_names.index(tagger)]

            else:
                y_pred = jets[
                    [
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                        for flav in class_labels
                    ]
                ].values

            rej_dict_tmp, disc_cut_dict_tmp = utt.GetRejection(
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
            for rej_type in rej_dict_tmp:
                tagger_rej_dicts[f"{tagger}_{rej_type}"].append(
                    rej_dict_tmp[rej_type]
                )

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
):
    """
    Get the probabilites in a new dict and calculate the discriminant scores.

    Input:
    - jets: Dataframe with the probabilites of the comparison taggers as columns
    - y_true: Internal truth labeling of the used jets.
    - tagger_preds: Prediction output of the taggers listed. e.g. [pred_dips, pred_umami]
    - tagger_names: Names of the freshly trained taggers. e.g. ["dips", "umami"]
    - tagger_list: List of the comparison tagger names.
    - class_labels: List of class labels which are used.
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - frac_values: Dict with the fraction values for the fresh taggers.
    - frac_values_comp: Dict with the fraction values for the comparison taggers.

    Output:
    - df_discs_dict: Dict with the discriminant scores of each jet and the probabilities
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
                    ] = jets[
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]}'
                    ]

                except KeyError:
                    logger.warning(
                        f'{tagger}_{flavour_categories[flav]["prob_var_name"]} is not in files! '
                        + "Skipping..."
                    )

        if tagger in tagger_names:
            y_pred = tagger_preds[counter]

        else:
            # Shape the probabilities of the comparison taggers like the output of the networks
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

            # Reshape to wrong sorted (transpose change it to correct shape)
            y_pred = tmp.reshape((len(class_labels_copy), -1))
            y_pred = np.transpose(y_pred)

        # Adding scores of the trained network
        df_discs_dict[f"disc_{tagger}"] = utt.GetScore(
            y_pred=y_pred,
            class_labels=class_labels_copy,
            main_class=main_class,
            frac_dict=frac_values[f"{tagger}"]
            if tagger in tagger_names
            else frac_values_comp[f"{tagger}"],
        )

    return df_discs_dict


def GetRejectionPerFractionDict(
    jets,
    tagger_preds: list,
    tagger_names: list,
    tagger_list: list,
    class_labels: list,
    main_class: str,
    target_beff: float,
    x_axis_granularity: int = 100,
    eff_min: float = 0.49,
    eff_max: float = 1.0,
):

    # TODO add fraction values scan
    logger.info("calculating rejections per fraction values")


def getDiscriminant(
    x,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
):
    """
    This method returns the score of the input (like GetScore)
    but calculated with the Keras Backend due to conflicts of
    numpy functions inside a layer in a keras model.
    """

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
    counter = x[:, index_dict[main_class]] + add_small

    # Calculate denominator of disc_score
    for class_label in class_labels_wo_main:
        denominator += frac_dict[class_label] * x[:, index_dict[class_label]]
    denominator += add_small

    return K.log(counter / denominator)


def discriminant_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    return (shape[0],)


def get_gradients(model, X, nJets):
    """
    Calculating the gradients with respect to the input variables.
    Note that only Keras backend functions can be used here because
    the gradients are tensorflow tensors and are not compatible with
    numpy.
    """
    gradients = K.gradients(model.output, model.inputs)

    input_tensors = model.inputs + [K.learning_phase()]
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # Pass in the cts and categorical inputs, as well as the learning phase
    # (0 for test mode)
    gradients = compute_gradients([X[:nJets], 0])

    return gradients[0]


def GetSaliencyMapDict(
    model,
    model_pred,
    X_test,
    Y_test,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    nJets: int = int(10e4),
):
    """
    Calculating the saliency maps dict.

    Input:
    - model: Loaded Keras model.
    - model_pred: Model predictions of the model.
    - Y_test: Truth labels in one-hot-encoded format.
    - class_labels: List of class labels which are used.
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - frac_dict: Dict with the fraction values for the tagger.

    Output:
    - Map_dict: Dict with the saliency values
    """

    logger.info("Calculate gradients for inputs")

    # Cut off last layer of the model for saliency maps
    cutted_model = model.layers[-1].output

    # Define the last node for the discriminant output
    disc = Lambda(
        getDiscriminant,
        output_shape=discriminant_output_shape,
        arguments={
            "class_labels": class_labels,
            "main_class": main_class,
            "frac_dict": frac_dict,
        },
    )(cutted_model)

    # Define the computation graph for the model
    model = Model(model.inputs, disc)

    # Define boolean mask to filter placeholder tracks
    boolMask = (np.sum(X_test, axis=-1) != 0).astype(bool)

    # Define the number of true tracks per jet as a mask
    nTrks = np.sum(boolMask, axis=-1)

    # Get score for the dips prediction
    Disc_values = utt.GetScore(
        y_pred=model_pred,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
    )

    # Init small dict
    map_dict = {}

    # Get spartial class id
    class_indices = [i for i in range(len(class_labels))]

    # Iterate over different beff, jet flavours and passed options
    for target_beff in [60, 70, 77, 85]:
        for (jet_flavour, class_index) in zip(class_labels, class_indices):
            for PassBool in [True, False]:

                # Get the Disc_values value for a specific flavour
                Disc_values_flavour = Disc_values[
                    Y_test[:, class_index].astype(bool)
                ]

                # Get the cutvalue for the specific WP
                cutvalue = np.percentile(
                    Disc_values_flavour, (100 - target_beff)
                )

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
                gradient_map = get_gradients(model, X_test[mask], nJets)

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
    model_tagger,
    main_class,
    model_frac_values,
    model_class_labels,
):
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
    return utt.GetScore(
        shaped_proba,
        class_labels=model_class_labels,
        main_class=main_class,
        frac_dict=model_frac_values,
    )
