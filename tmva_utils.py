import ROOT
import uproot
import numpy as np
from array import array
import yaml
from os import makedirs
from os.path import join, isfile
from sklearn.utils.class_weight import compute_class_weight
from utils import generate_cv_data


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


def train_tmva_model(data, model_identifier="BDT", suffix_string="test",
                     root_file_dir="TMVA_models/"):

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]

    # converting to ROOT TTrees
    ens_root_file_dir = join(root_file_dir, suffix_string)
    makedirs(ens_root_file_dir, exist_ok=True)
    tree_path = join(ens_root_file_dir, f"trees_{suffix_string}.root")
    tree_file = uproot.recreate(tree_path)
    tree_file["sig_train"] = array_to_dict(x_train[y_train == 1])
    tree_file["bkg_train"] = array_to_dict(x_train[y_train == 0])
    tree_file["sig_val"] = array_to_dict(x_val[y_val == 1])
    tree_file["bkg_val"] = array_to_dict(x_val[y_val == 0])

    # loading them again from file to convert uproot trees to native TTrees
    tree_file_root = ROOT.TFile(tree_path, "READ")
    tree_sig_train = tree_file_root.Get("sig_train")
    tree_bkg_train = tree_file_root.Get("bkg_train")
    tree_sig_val = tree_file_root.Get("sig_val")
    tree_bkg_val = tree_file_root.Get("bkg_val")

    class_weights_train = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    class_weights_val = compute_class_weight(
        'balanced', classes=np.unique(y_val), y=y_val)

    # instantiating a TMVA factory
    model_data_path = join(ens_root_file_dir, f"data_{suffix_string}.root")

    ROOT.TMVA.Tools.Instance()
    model_data_file = ROOT.TFile(model_data_path, "RECREATE")
    factory = ROOT.TMVA.Factory("TMVAClassification", model_data_file,
                                ":".join(
                                    ["!V",
                                     "!Silent",
                                     "Color",
                                     "DrawProgressBar",
                                     "Transformations=I;D;P;G,D",
                                     "AnalysisType=Classification"
                                     ]))

    # putting data into TMVA dataloader
    dataloader = ROOT.TMVA.DataLoader(ens_root_file_dir)
    for i in range(x_train.shape[1]):
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

    return model_data_path


def train_tmva_ensemble(data, num_models=10, cv_mode="fixed",
                        model_identifier="BDT",
                        root_file_dir="TMVA_models/"):
    """
    Trains an ensemble of default TMVA BDT models and returns the
    mean predictions on the test set.

    Args:
        data (dict): A dictionary containing the training, validation and test
            sets as well as the corresponding labels.
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
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir (str, optional): The directory to save the ROOT files.

    Returns:
        list: A list of paths to the model root files.

    """

    model_list = []

    cv_data = generate_cv_data(data, num_models, cv_mode)
    for ens, dat in zip(range(num_models), cv_data):
        print(f"Training model {ens+1}/{num_models}...")

        tmp_model_data_path = train_tmva_model(
            dat, model_identifier=model_identifier, suffix_string=f"ens{ens}",
            root_file_dir=root_file_dir)

        model_list.append(tmp_model_data_path)

    return model_list


def train_tmva_multi(data, num_runs=10, ensembles_per_model=10,
                     cv_mode="fixed", model_identifier="BDT",
                     root_file_dir_base="tmva_root_files"):
    """
    Run multiple ensembles of default TMVA BDT trainings and
    return array of mean test predictions for each ensemble.

    Args:
        data (dict): A dictionary containing the training, validation and test
            sets as well as the corresponding labels.
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
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir_base (str, optional): The base name of the directory
            where ROOT files will be stored. A run number will be appended.

    Returns:
        list: A list of list of model root file paths. The first index of the
            list corresponds to the run number, the second index to the
            ensemble number.
    """
    if cv_mode not in ["fixed", "random", "k-fold"]:
        raise ValueError(
            "cv_mode must be either 'fixed', 'random' or 'k-fold'"
            )

    run_models = []

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        run_dir = root_file_dir_base+f"_{run}"
        makedirs(run_dir, exist_ok=True)

        ens_models = train_tmva_ensemble(
            data, num_models=ensembles_per_model, cv_mode=cv_mode,
            root_file_dir=run_dir, model_identifier=model_identifier)

        run_models.append(ens_models)

    return run_models


def eval_tmva_model(data, model_identifier="BDT",
                    root_file_dir="TMVA_models/",
                    suffix_string="test"):
    """Evaluate a single TMVA BDT model on the test set.

    Args:
        data (dict): A dictionary containing the training, validation and test
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir (str, optional): The directory where the ROOT files
            containing the trained model is stored. Defaults to
            "TMVA_models/".
        suffix_string (str, optional): The suffix string to append to the
            model name. Used e.g. to denote an ensemble model number.
            Defaults to "test".

    Returns:
        A flat numpy array containing the predictions of the TMVA BDT
            model on the test set.
    """

    x_test = data["x_test"]

    # convert data to native TTree
    tree_path = join(root_file_dir, f"test_tree_{suffix_string}.root")
    tree_file = uproot.recreate(tree_path)
    tree_file["test"] = array_to_dict(x_test)
    tree_file_root = ROOT.TFile(tree_path, "READ")
    tree_test = tree_file_root.Get("test")

    # compute model predictions on the test set
    reader = ROOT.TMVA.Reader()
    var_names = [f"var{i}" for i in range(x_test.shape[1])]
    vars = []
    for var_name in var_names:
        vars.append(array('f', [0]))
        reader.AddVariable(var_name, vars[-1])

    tmp_model_preds = np.ones(x_test.shape[0]) * -999.
    reader.BookMVA(
        model_identifier,
        join(root_file_dir, "weights",
             f"TMVAClassification_{model_identifier}.weights.xml")
    )

    for i in range(len(var_names)):
        tree_test.SetBranchAddress(var_names[i], vars[i])

    for evt in range(tree_test.GetEntries()):
        tree_test.GetEntry(evt)
        tmp_model_preds[evt] = reader.EvaluateMVA(model_identifier)

    tree_file_root.Close()

    return tmp_model_preds


def eval_tmva_ensemble(data, num_models=10,
                       model_identifier="BDT",
                       root_file_dir="TMVA_models/",
                       save_full_preds=None):
    """Evaluate an ensemble of TMVA BDT models on the test set.

    Args:
        data (dict): A dictionary containing the training, validation and test
        num_models (int, optional): The number of models in the ensemble.
            Defaults to 10.
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir (str, optional): The directory where the ROOT files
            containing the trained models are stored. Defaults to
            "TMVA_models/".
        save_full_preds (str, optional): If not None, the full predictions of
            each model in the ensemble will be saved to the specified path.

    Returns:
        A flat numpy array containing the mean predictions of each TMVA BDT
            ensemble on the test set.
    """

    ens_preds_list = []
    for ens in range(num_models):
        ens_root_file_dir = join(root_file_dir, f"ens{ens}")
        tmp_model_preds = eval_tmva_model(
                          data,
                          model_identifier=model_identifier,
                          root_file_dir=ens_root_file_dir,
                          suffix_string=f"ens{ens}")
        ens_preds_list.append(tmp_model_preds)

    ens_preds = np.stack(ens_preds_list, axis=0)
    if save_full_preds is not None:
        print(f"Saving full predictions as {save_full_preds}")
        np.save(save_full_preds, ens_preds)

    return np.mean(ens_preds, axis=0)


def eval_tmva_multi(data, num_runs=10, ensembles_per_model=10,
                    model_identifier="BDT",
                    root_file_dir_base="tmva_root_files",
                    save_ensemble_preds=False):
    """Evaluate multiple ensembles of TMVA BDT models on the test set.

    Args:
        data (dict): A dictionary containing the training, validation and test
        num_runs (int, optional): The number of full ensembles to train.
            Defaults to 10.
        ensembles_per_model (int, optional): The number of models trained
            per ensemble. Defaults to 10.
        model_identifier (str, optional): The identifier to use for the
            TMVA model configuration. It must coincide with the name of a
            YAML file in ./TMVA_configs/. Defaults to "BDT".
        root_file_dir_base (str, optional): The base name of the directory
            where ROOT files will be stored. A run number will be appended.
            Defaults to "tmva_root_files".
        save_ensemble_preds (bool, optional): If True, the full predictions of
            each ensemble will be saved to a .npy file.

    Returns:
        full_preds (array-like): The mean predictions of each ensemble on
            the test set, with shape (num_runs, x_test.shape[0]).
    """

    full_preds_list = []

    for run in range(num_runs):
        run_dir = root_file_dir_base+f"_{run}"
        if save_ensemble_preds:
            save_str = join(run_dir, f"ensemble_preds_run{run}.npy")
        else:
            save_str = None

        ens_mean_preds = eval_tmva_ensemble(
            data, num_models=ensembles_per_model,
            model_identifier=model_identifier,
            root_file_dir=run_dir, save_full_preds=save_str)

        full_preds_list.append(ens_mean_preds)

    return np.stack(full_preds_list, axis=0)
