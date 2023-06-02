import ROOT
import uproot
import numpy as np
from os import makedirs
from os.path import join, isfile
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from array import array
import yaml


def array_to_dict(input_array):
    """
    Converts a numpy ndarray to a dictionary of arrays, where each key
    is "var{i}" with i being the first dimensional index of the variable
    in the array.

    Args:
        input_array (ndarray): The input array.

    Returns:
        dict: The dictionary of arrays.
    """
    out_dict = {}
    for i in range(input_array.shape[1]):
        out_dict[f"var{i}"] = input_array[:, i]
    return out_dict


def import_settings(yaml_file):
    """
    Imports the model settings from a YAML file and returns them as a
    string that can be used as input for TMVA.Factory.BookMethod.

    Args:
        yaml_file (str): The path to the YAML file.

    Returns:
        str: The settings as a string.
    """
    with open(yaml_file, 'r') as stream:
        settings = yaml.safe_load(stream)
    parsed_settings = []
    for key, value in settings.items():
        if isinstance(value, bool) and value:
            parsed_settings.append(key)
        else:
            parsed_settings.append(f"{key}={value}")
    return ":".join(parsed_settings)


def train_tmva_ensemble(x_train, y_train, x_val, y_val, x_test,
                        y_test, num_models=10, cv_mode="fixed",
                        save_full_preds=None,
                        model_identifier="BDT",
                        root_file_dir="TMVA_models/"):
    """
    Trains an ensemble of default TMVA BDT models and returns the
    mean predictions on the test set.

    Args:
        x_train (ndarray): The training features.
        y_train (ndarray): The training set labels.
        x_val (ndarray): The validation set features.
        y_val (ndarray): The validation set labels.
        x_test (ndarray): The test set features.
        y_test (ndarray): The test set labels.
        num_models (int, optional): The number of models in the ensemble.
            Defaults to 10.
        cv_mode (str, optional): The cross-validation mode to use. Valid values
            are "fixed", "random", or "k-fold". The meaning of the available
            modes is as follows:

            - "fixed": Train ensemble on a fixed assignment of training and
              validation set.
            - "random": Concatenate training and validation set and randomly
              assign training and validation samples for each model
              constituting the ensemble
            - "k-fold": Concatenate training and validation set, then split
              data into `num_models` equally sized parts assign one fold as
              validation set  and the remaining folds as training set. Train
              all possible assignments (i.e. you should end up with
              `num_models` models each trained on a different train/validation
              k-fold assignment)

            Defaults to "fixed".
        save_full_preds (str, optional): The filename to save the full
            predictions of the ensemble on the test set as a .npy file.
            Defaults to None (in which case no predictions will be saved).
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir (str, optional): The directory to save the ROOT files.

    Returns:
        array: The mean predictions on the test set.

    """

    if cv_mode == "k-fold":

        x_full = np.vstack([x_train, x_val])
        y_full = np.hstack([y_train, y_val])
        x_split = np.array_split(x_full, num_models)
        y_split = np.array_split(y_full, num_models)

    for ens in range(num_models):
        print(f"Training model {ens+1}/{num_models}...")

        if cv_mode == "random":
            x_full = np.vstack([x_train, x_val])
            y_full = np.hstack([y_train, y_val])
            x_train_tmp, x_val_tmp, y_train_tmp, y_val_tmp = train_test_split(
                x_full, y_full, test_size=0.5
                )

            x_test_tmp = x_test

        elif cv_mode == "k-fold":
            x_val_tmp = x_split[ens]
            y_val_tmp = y_split[ens]
            x_train_tmp = np.concatenate(
                [x_split[i] for i in range(num_models) if i != ens]
                )

            y_train_tmp = np.concatenate(
                [y_split[i] for i in range(num_models) if i != ens]
                )

            x_test_tmp = x_test

        else:
            x_train_tmp = x_train
            y_train_tmp = y_train
            x_val_tmp = x_val
            y_val_tmp = y_val
            x_test_tmp = x_test

        # converting to ROOT TTrees
        ens_root_file_dir = join(root_file_dir, f"ens{ens}")
        makedirs(ens_root_file_dir, exist_ok=True)
        tree_path = join(ens_root_file_dir, f"trees_ens{ens}.root")
        tree_file = uproot.recreate(tree_path)
        tree_file["sig_train"] = array_to_dict(x_train_tmp[y_train_tmp == 1])
        tree_file["bkg_train"] = array_to_dict(x_train_tmp[y_train_tmp == 0])
        tree_file["sig_val"] = array_to_dict(x_val_tmp[y_val_tmp == 1])
        tree_file["bkg_val"] = array_to_dict(x_val_tmp[y_val_tmp == 0])
        tree_file["test"] = array_to_dict(x_test_tmp)

        # loading them again from file to convert uproot trees to native TTrees
        tree_file_root = ROOT.TFile(tree_path, "READ")
        tree_sig_train = tree_file_root.Get("sig_train")
        tree_bkg_train = tree_file_root.Get("bkg_train")
        tree_sig_val = tree_file_root.Get("sig_val")
        tree_bkg_val = tree_file_root.Get("bkg_val")
        tree_test = tree_file_root.Get("test")

        class_weights_train = compute_class_weight(
            'balanced', classes=np.unique(y_train_tmp), y=y_train_tmp)
        class_weights_val = compute_class_weight(
            'balanced', classes=np.unique(y_val_tmp), y=y_val_tmp)

        # instantiating a TMVA factory
        ROOT.TMVA.Tools.Instance()
        model_data_path = join(ens_root_file_dir, f"data_ens{ens}.root")
        model_data_file = ROOT.TFile(model_data_path, "RECREATE")
        factory = ROOT.TMVA.Factory("TMVAClassification", model_data_file,
                                    ":".join(["!V",
                                              "!Silent",
                                              "Color",
                                              "DrawProgressBar",
                                              "Transformations=I;D;P;G,D",
                                              "AnalysisType=Classification"
                                              ]))

        # putting data into TMVA dataloader
        dataloader = ROOT.TMVA.DataLoader(ens_root_file_dir)
        for i in range(x_train_tmp.shape[1]):
            dataloader.AddVariable(f"var{i}", "F")
        dataloader.AddBackgroundTree(tree_bkg_train,
                                     class_weights_train[0],
                                     "Training")
        dataloader.AddBackgroundTree(tree_bkg_val,
                                     class_weights_val[0],
                                     "Test")
        dataloader.AddSignalTree(tree_sig_train,
                                 class_weights_train[1],
                                 "Training")
        dataloader.AddSignalTree(tree_sig_val,
                                 class_weights_val[1],
                                 "Test")

        # defining BDT model and adding dataloader
        config_file = join(f"TMVA_configs/{model_identifier}.yml")
        assert isfile(config_file), f"Config file {config_file} not found."
        factory.BookMethod(dataloader,
                           ROOT.TMVA.Types.kBDT,
                           model_identifier,
                           f"!H:!V:{import_settings(config_file)}")

        # actual training
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
        model_data_file.Close()

        # compute model predictions on the test set
        reader = ROOT.TMVA.Reader()
        var_names = [f"var{i}" for i in range(x_train_tmp.shape[1])]
        vars = []
        for var_name in var_names:
            vars.append(array('f', [0]))
            reader.AddVariable(var_name, vars[-1])

        tmp_model_preds = np.ones_like(y_test) * -999.
        reader.BookMVA(
            model_identifier,
            join(ens_root_file_dir,
                 "weights",
                 f"TMVAClassification_{model_identifier}.weights.xml")
        )

        for i in range(len(var_names)):
            tree_test.SetBranchAddress(var_names[i], vars[i])

        for evt in range(tree_test.GetEntries()):
            tree_test.GetEntry(evt)
            tmp_model_preds[evt] = reader.EvaluateMVA(model_identifier)

        # For each model in the ensemble, stack the test predictions
        if ens == 0:
            ens_preds = tmp_model_preds.reshape(1, -1)
        else:
            ens_preds = np.vstack([ens_preds, tmp_model_preds])

        tree_file_root.Close()

    if save_full_preds is not None:
        print(f"Saving full predictions as {save_full_preds}")
        np.save(save_full_preds, ens_preds)

    # Finally, take mean of all predictions in the ensemble
    ens_mean_preds = np.mean(ens_preds, axis=0)

    return ens_mean_preds


def train_tmva_multi(x_train, y_train, x_val, y_val, x_test, y_test,
                     num_runs=10, ensembles_per_model=10,
                     cv_mode="fixed",
                     save_ensemble_preds=False,
                     model_identifier="BDT",
                     root_file_dir_base="tmva_root_files"):
    """
    Run multiple ensembles of default TMVA BDT trainings and
    return array of mean test predictions for each ensemble.

    Args:
        x_train (array-like): The training data to fit the model.
        y_train (array-like): The target values for the training data.
        x_val (array-like): The validation data to fit the model.
        y_val (array-like): The target values for the validation data.
        x_test (array-like): The test data to predict using the trained models.
        y_test (array-like): The true labels for the test data.
        num_runs (int, optional): The number of ensemble trainings to run.
            Default is 10.
        ensembles_per_model (int, optional): The number of ensembles to train
            per ensemble. Default is 10.
        cv_mode (str, optional): The cross-validation mode to use. Valid values
            are "fixed", "random", or "k-fold". The meaning of the available
            modes is as follows:

            - "fixed": Train ensemble on a fixed assignment of training and
              validation set.
            - "random": Concatenate training and validation set and randomly
              assign training and validation samples for each model
              constituting the ensemble
            - "k-fold": Concatenate training and validation set, then split
              data into `num_models` equally sized parts assign one fold as
              validation set  and the remaining folds as training set. Train
              all possible assignments (i.e. you should end up with
              `num_models` models each trained on a different train/validation
              k-fold assignment)

            Defaults to "fixed".
        save_ensemble_preds (bool, optional): Whether or not to save the full
            ensemble predictions during training. Default is False.
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir_base (str, optional): The base name of the directory
            where ROOT files will be stored. A run number will be appended.

    Returns:
        full_preds (array-like): The mean predictions of each ensemble on
            the test set, with shape (num_runs, x_test.shape[0]).
    """
    if cv_mode not in ["fixed", "random", "k-fold"]:
        raise ValueError(
            "cv_mode must be either 'fixed', 'random' or 'k-fold'"
            )

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        run_dir = root_file_dir_base+f"_{run}"
        makedirs(run_dir, exist_ok=True)
        if save_ensemble_preds:
            save_str = join(run_dir, f"ensemble_preds_run{run}.npy")
        else:
            save_str = None

        ens_mean_preds = train_tmva_ensemble(
            x_train, y_train, x_val, y_val, x_test, y_test,
            num_models=ensembles_per_model, cv_mode=cv_mode,
            save_full_preds=save_str, root_file_dir=run_dir,
            model_identifier=model_identifier
            )

        ens_mean_preds = ens_mean_preds.reshape((1, -1))

        if run == 0:
            full_preds = ens_mean_preds
        else:
            full_preds = np.vstack([full_preds, ens_mean_preds])

    return full_preds
