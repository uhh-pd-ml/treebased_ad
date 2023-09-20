import numpy as np
from os.path import join, exists, isdir
from os import makedirs, listdir
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier,
                              AdaBoostClassifier)
from torch_DNN import PyTorchClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from copy import deepcopy
import pandas as pd
import torch


class HGBPipeline(Pipeline):
    """A scikit-learn Pipeline extension for histogrammed gradient boosting.

    This class extends the Pipeline class from scikit-learn to include support
    for the `HistGradientBoosting` models and their specific prediction method.

    Attributes:
        steps (list): A list of tuples containing the name and transformation
            object for each step in the pipeline.
        memory (str, joblib.Memory, None): The caching strategy to use for
            transformers that have caching enabled.
        verbose (bool): A boolean indicating whether to print progress messages
            during fitting.

    """

    def __init__(self, steps, *, memory=None, verbose=False):
        """Initializes the HGBPipeline object.

        Args:
            steps (list): A list of tuples containing the name and
                transformation object for each step in the pipeline.
            memory (str, joblib.Memory, None): The caching strategy to use
                for transformers that have caching enabled.
            verbose (bool): A boolean indicating whether to print progress
                messages during fitting.

        """
        super().__init__(steps, memory=memory, verbose=verbose)

    def staged_predict_proba(self, X, **predict_proba_params):
        """Apply pipeline and predict class probabilities at each iteration.

        Args:
            X (array-like or sparse matrix): Input data to be transformed and
                used for prediction.
            **predict_proba_params (dict): Additional keyword arguments to be
                passed to the staged_predict_proba method of the final
                estimator in the pipeline.

        Returns:
            An iterator over the predicted class probabilities for each stage
            of the gradient boosting model.

        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].staged_predict_proba(Xt,
                                                      **predict_proba_params)


def load_single_model(model_dir, run_num, model_num):
    """Load a single model from a directory.

    Args:
        model_dir (str): The parent directory containing the sub-directory
                         for the specified run, which then should contain the
                         model file to load.
        run_num (int): The run number of the model to load.
        model_num (int): The model number of the model to load.

    Returns:
        The loaded model.

    """
    return joblib.load(
        join(model_dir, f"run_{run_num}", f"model_{model_num}.joblib"))


def load_models_allruns(model_dir, is_dnn=False):
    """Load all models from a directory.

    Args:
        model_dir (str): The directory containing the models to load.
                         It is required that the structure of the
                         sub-directories is such that they start with
                         "run_" followed by the run number and contain the
                         model files named "model_0.joblib", "model_1.joblib",
                         etc.

    Returns:
        A list of the loaded models.

    """

    if is_dnn:
        load_func = load_single_dnn_model
    else:
        load_func = load_single_model

    num_runs = 0
    for tmp_dir in listdir(model_dir):
        if tmp_dir.startswith("run_") and isdir(join(model_dir, tmp_dir)):
            num_runs += 1

    if num_runs == 0:
        raise ValueError("No runs found in model directory!")

    all_models = []
    for i in range(num_runs):
        # Assume number of models in an ensemble are the same for each run
        if i == 0:
            models_per_run = 0
            for tmp_dir in listdir(join(model_dir, f"run_{i}")):
                if tmp_dir.endswith("joblib"):
                    models_per_run += 1

        tmp_model_list = []
        for j in range(models_per_run):
            tmp_model_list.append(load_func(model_dir, i, j))

        all_models.append(tmp_model_list)

    return all_models


def save_model(model, save_dir, model_num):
    """Save a trained model to a file.

    Args:
        model: The trained model to save.
        save_dir (str): The directory to save the model to.
        model_num (int): The model number to use in the filename.

    """
    joblib.dump(model, join(save_dir, f"model_{model_num}.joblib"))


def save_dnn_model(model, save_dir, model_num):
    """Save a trained DNN model to a file.

    Args:
        model: The trained model to save.
        save_dir (str): The directory to save the model to.
        model_num (int): The model number to use in the filename.

    """

    # save scaler
    scaler = model["scaler"]
    joblib.dump(scaler, join(save_dir, f"scaler_{model_num}.joblib"))

    # save model state dict
    torch.save(model["clsf"].model.state_dict(),
               join(save_dir, f"model_{model_num}.pt"))

    np.save(join(save_dir, f"val_losses_{model_num}.npy"),
            np.array(model["clsf"].val_losses),
            )
    np.save(join(save_dir, f"train_losses_{model_num}.npy"),
            np.array(model["clsf"].train_losses),
            )


def load_single_dnn_model(model_dir, run_num, model_num):
    """Load a trained DNN model from a file.

    Args:
        model_dir (str): The directory containing the model to load.
        model_num (int): The model number of the model to load.

    Returns:
        The loaded model.

    """

    # load scaler
    scaler = joblib.load(join(model_dir, f"run_{run_num}",
                              f"scaler_{model_num}.joblib"))

    n_inputs = scaler.n_features_in_
    # load model state dict
    model_state_dict = torch.load(join(model_dir, f"run_{run_num}",
                                       f"model_{model_num}.pt"))

    # load model
    py_model = PyTorchClassifier(
        layers=[64, 64, 64],
        validation_fraction=0.5,
        split_seed=42,
        input_size=n_inputs,
        )

    py_model.model.load_state_dict(model_state_dict)
    pipe = HGBPipeline([("scaler", scaler), ("clsf", py_model)])

    pipe.val_losses = np.load(join(model_dir, f"run_{run_num}",
                                   f"val_losses_{model_num}.npy"))

    pipe.train_losses = np.load(join(model_dir, f"run_{run_num}",
                                     f"train_losses_{model_num}.npy"))

    return pipe


def load_lhco_rd_moreinputs(data_dir, inputs, shuffle=False):
    input_categories = {
        "vanilla": ["mj2", "delta_mj", "tau21j1_1", "tau21j2_1"],
        "plain_subjettinesses_1": [
            f"tau{i}j{j}_1" for i in range(1, 10) for j in range(1, 3)
        ],
        "plain_subjettinesses_2": [
            f"tau{i}j{j}_2" for i in range(1, 10) for j in range(1, 3)
        ],
        "plain_subjettinesses_5": [
            f"tau{i}j{j}_5" for i in range(1, 10) for j in range(1, 3)
        ],
        "tau_ratios_1": [
            f"tau{i}{i-1}j{j}_1" for i in range(2, 10) for j in range(1, 3)
        ],
        "tau_ratios_2": [
            f"tau{i}{i-1}j{j}_2" for i in range(2, 10) for j in range(1, 3)
        ],
        "tau_ratios_5": [
            f"tau{i}{i-1}j{j}_5" for i in range(2, 10) for j in range(1, 3)
        ],
        "tau_x1_1": [
            f"tau{i}1j{j}_1" for i in range(2, 10) for j in range(1, 3)
        ],
        "tau_x1_2": [
            f"tau{i}1j{j}_2" for i in range(2, 10) for j in range(1, 3)
        ],
        "tau_x1_5": [
            f"tau{i}1j{j}_5" for i in range(2, 10) for j in range(1, 3)
        ]
    }

    input_list = []

    for inp in inputs:
        if inp in input_categories.keys():
            input_list += input_categories[inp]
        else:
            input_list.append(inp)

    # remove duplicates
    input_list = sorted(list(set(input_list)))

    input_list.append("label")
    data_train = pd.read_hdf(
        join(data_dir, "innerdata_train.h5"), key="df"
    )[input_list]

    data_val = pd.read_hdf(
        join(data_dir, "innerdata_val.h5"), key="df"
    )[input_list]
    data_test = pd.read_hdf(
        join(data_dir, "innerdata_test.h5"), key="df"
    )[input_list]
    bg_train = pd.read_hdf(
        join(data_dir, "innerdata_extrabkg_train.h5"), key="df"
    )[input_list]
    bg_val = pd.read_hdf(
        join(data_dir, "innerdata_extrabkg_val.h5"), key="df"
    )[input_list]

    X_train = np.concatenate(
        [data_train.drop("label", axis=1).values,
         bg_train.drop("label", axis=1).values])

    X_val = np.concatenate(
        [data_val.drop("label", axis=1).values,
         bg_val.drop("label", axis=1).values])

    X_test = data_test.drop("label", axis=1).values
    y_train_sigbg = np.concatenate(
        [data_train["label"].values,
         bg_train["label"].values])

    y_train_databg = np.concatenate(
        [np.ones(len(data_train)),
         np.zeros(len(bg_train))])

    y_val_sigbg = np.concatenate(
        [data_val["label"].values,
         bg_val["label"].values])

    y_val_databg = np.concatenate(
        [np.ones(len(data_val)),
         np.zeros(len(bg_val))])

    y_test = data_test["label"].values

    return {"x_train": X_train, "y_train_databg": y_train_databg,
            "y_train_sigbg": y_train_sigbg, "x_val": X_val,
            "y_val_databg": y_val_databg, "y_val_sigbg": y_val_sigbg,
            "x_test": X_test, "y_test": y_test}


def load_lhco_rd(data_dir, shuffle=False):
    """Load the LHCO R&D dataset.

    This function loads the LHCO R&D dataset from a specified directory.
    The dataset consists of 3 sets: training set, validation set and test set.
    For the training and validation set, the data/background labels are
    needed, while for the test set, signal/background labels are required. The
    function performs the following steps:

        1. Loads the training set, validation set and testset including their
           respective "extra" background samples.
        2. Concatenates original and extra background samples for each dataset.
        3. Shuffles the training and validation sets (if shuffle is set to
           True).
        4. Converts the data to the float32 data type (needed for
           pytorch DNN classifier training).

    Args:
        data_dir (str): The path to the directory containing the LHCO dataset.
        shuffle (bool, optional): Whether to shuffle the training and
            validation sets. Default is False.

    Returns:
        dict: A dictionary containing the training, validation and test sets
            as well as the corresponding labels.
    """

    # for train and val set, we only need data/bg labels
    X_train = np.load(join(data_dir, "innerdata_train.npy"))[:, 1:-1]
    X_train_extrabg = np.load(
        join(data_dir, "innerdata_extrabkg_train.npy")
        )[:, 1:-1]

    y_train_databg = np.concatenate((np.ones((X_train.shape[0], )),
                                     np.zeros((X_train_extrabg.shape[0], ))))

    y_train_sigbg = np.concatenate(
        (np.load(join(data_dir, "innerdata_train.npy"))[:, -1],
         np.zeros((X_train_extrabg.shape[0], )))
        )

    X_train = np.concatenate((X_train, X_train_extrabg))

    X_val = np.load(join(data_dir, "innerdata_val.npy"))[:, 1:-1]
    X_val_extrabg = np.load(
        join(data_dir, "innerdata_extrabkg_val.npy")
        )[:, 1:-1]

    y_val_databg = np.concatenate((np.ones((X_val.shape[0], )),
                                   np.zeros((X_val_extrabg.shape[0], ))))

    y_val_sigbg = np.concatenate(
        (np.load(join(data_dir, "innerdata_val.npy"))[:, -1],
         np.zeros((X_val_extrabg.shape[0], )))
        )

    X_val = np.concatenate((X_val, X_val_extrabg))

    if shuffle:
        # shuffle train set
        shuffle_arr = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle_arr)
        X_train = X_train[shuffle_arr]
        y_train_databg = y_train_databg[shuffle_arr]
        y_train_sigbg = y_train_sigbg[shuffle_arr]

        # shuffle val set
        shuffle_arr_val = np.arange(X_val.shape[0])
        np.random.shuffle(shuffle_arr_val)
        X_val = X_val[shuffle_arr_val]
        y_val_databg = y_val_databg[shuffle_arr_val]
        y_val_sigbg = y_val_sigbg[shuffle_arr_val]

    # for test set, we only need sig/bg labels
    X_test = np.load(join(data_dir, "innerdata_test.npy"))[:, 1:-1]
    y_test = np.load(join(data_dir, "innerdata_test.npy"))[:, -1]
    X_test_extrabg = np.load(
        join(data_dir, "innerdata_extrabkg_test.npy")
        )[:, 1:-1]

    y_test_extrabg = np.zeros((X_test_extrabg.shape[0], ))
    X_test = np.concatenate((X_test, X_test_extrabg))
    y_test = np.concatenate((y_test, y_test_extrabg))

    # convert to float32 (useful for pytorch DNN training)
    X_train = X_train.astype(np.float32)
    y_train_databg = y_train_databg.astype(np.float32)
    y_train_sigbg = y_train_sigbg.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val_databg = y_val_databg.astype(np.float32)
    y_val_sigbg = y_val_sigbg.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return {"x_train": X_train, "y_train_databg": y_train_databg,
            "y_train_sigbg": y_train_sigbg, "x_val": X_val,
            "y_val_databg": y_val_databg, "y_val_sigbg": y_val_sigbg,
            "x_test": X_test, "y_test": y_test}


def add_gaussian_features(data, n_gaussians=10):
    """Adds Gaussian-distributed random variables to input features

    Args:
        data (dict): Dictionary containing the training, validation and test
            sets as well as the corresponding labels.
        n_gaussians (int, optional): Number of Gaussian random variables to
            generate and add to the input data. Default is 10.

    Returns:
        data (dict): Dictionary containing the training, validation and test
            sets as well as the corresponding labels, with Gaussian random
            variables added to the input data.
    """
    new_data = deepcopy(data)
    rand_var_train = np.random.randn(new_data["x_train"].shape[0], n_gaussians)
    rand_var_val = np.random.randn(new_data["x_val"].shape[0], n_gaussians)
    rand_var_test = np.random.randn(new_data["x_test"].shape[0], n_gaussians)

    new_data["x_train"] = np.hstack(
        [new_data["x_train"], rand_var_train]
        ).astype(np.float32)

    new_data["x_val"] = np.hstack(
        [new_data["x_val"], rand_var_val]
        ).astype(np.float32)

    new_data["x_test"] = np.hstack(
        [new_data["x_test"], rand_var_test]
        ).astype(np.float32)

    return new_data


def multi_roc_sigeffs(preds, labels):
    """Compute TPRs and FPRs for multiple predictions using common TPR values.

    Args:
        preds: A numpy array of shape (n_runs, n_samples) representing the
            predicted probabilities on a dataset for each run.
        labels: A numpy array of shape (n_samples,) representing the
            class labels (0 or 1) for each sample in preds.

    Returns:
        A tuple containing two numpy arrays:
            - tprs: A numpy array of shape (n_runs, n_thresholds)
                    representing the true positive rates for each threshold
                    value.
            - fprs: A numpy array of shape (n_runs, n_thresholds) representing
                    the false positive rates for each threshold value.

    """
    sig_preds = preds[:, labels == 1]
    bg_preds = preds[:, labels == 0]
    tprs = np.linspace(0, 1, 1000)
    for idx, tpr in enumerate(tprs):
        thresh = np.quantile(sig_preds, 1-tpr, axis=1).reshape((-1, 1))
        if idx == 0:
            fprs = np.sum(bg_preds > thresh, axis=1)/bg_preds.shape[1]
            fprs = fprs.reshape(preds.shape[0], -1)
        else:
            tmp_fprs = np.sum(bg_preds > thresh, axis=1)/bg_preds.shape[1]
            tmp_fprs = tmp_fprs.reshape(preds.shape[0], -1)
            fprs = np.hstack((fprs, tmp_fprs))

    tprs = np.tile(tprs, (preds.shape[0], 1))

    return tprs, fprs


def get_losses(hist_model, x, y, compute_weights=False):

    if compute_weights:
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y), y=y
            )

        sample_weights = ((np.ones(y.shape) - y)*class_weights[0]
                          + y*class_weights[1])

    else:
        sample_weights = None

    losses = []
    preds_gen = hist_model.staged_predict_proba(x)
    for preds in preds_gen:
        losses.append(log_loss(y, preds[:, 1], sample_weight=sample_weights))

    return np.array(losses)


def preds_from_optimal_iter(hist_model, x_val, y_val, x_test,
                            compute_weights=True):
    """Get test predictions of optimal iteration based on min. val loss.

    Args:
        hist_model: a trained tree-based model.
        x_val: an array-like matrix of features of the validation set.
        y_val: an array-like matrix of labels of the validation set.
        x_test: an array-like matrix of features of the test set.
        compute_weights: a bool indicating whether to compute the weighted
            validation loss or not. Default is True.

    Returns:
        An array of predictions for the test set at the iteration with the
        minimum validation loss.

    """
    weighted_val_losses = get_losses(hist_model, x_val, y_val,
                                     compute_weights=True)

    unweighted_val_losses = get_losses(hist_model, x_val, y_val,
                                       compute_weights=False)

    if compute_weights:
        val_losses = weighted_val_losses
    else:
        val_losses = unweighted_val_losses

    best_iter = np.argmin(val_losses)
    print((f"Best iteration: {best_iter+1}, "
           f"unweighted val loss: {unweighted_val_losses[best_iter]:.4f}, "
           f"weighted val loss: {weighted_val_losses[best_iter]:.4f}"))

    test_preds_gen = hist_model.staged_predict_proba(x_test)
    for i, test_preds in enumerate(test_preds_gen):
        if i == best_iter:
            return test_preds[:, 1]


def get_sample_weights(y):
    # Compute class weights for training set for weighted loss
    class_weights = class_weight.compute_class_weight(
                class_weight='balanced', classes=np.unique(y),
                y=y
                )

    sample_weights = (
        (np.ones(y.shape) - y)*class_weights[0]
        + y*class_weights[1]
        )

    return sample_weights


def train_rf_model(data, max_iters=100):

    clsf_hist_model = RandomForestClassifier(
            n_estimators=max_iters, max_depth=23, min_samples_leaf=20,
            max_features="log2", min_samples_split=14,
            max_samples=0.5434762030454895, class_weight="balanced",
            )

    steps = [('scaler', StandardScaler()), ('clsf', clsf_hist_model)]

    tmp_hist_model = HGBPipeline(steps)

    tmp_hist_model.fit(data["x_train"], data["y_train"])

    # Save seed for random split so train/val split can be reproduced
    if "split_val" in data.keys():
        tmp_hist_model.split_seed = data["split_val"]

    tmp_hist_model.val_losses = None
    tmp_hist_model.train_losses = None

    return tmp_hist_model


def train_ada_model(data, early_stopping=True,
                    max_iters=100):

    # Scikit-learn's AdaBoostClassifier does not support early stopping
    if early_stopping:
        raise NotImplementedError
    else:
        clsf_hist_model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=None, max_leaf_nodes=31, min_samples_leaf=20,
                    class_weight='balanced'
                    ),
                n_estimators=max_iters, learning_rate=0.1
                )

        steps = [('scaler', StandardScaler()), ('clsf', clsf_hist_model)]

        tmp_hist_model = HGBPipeline(steps)
        tmp_hist_model.fit(data["x_train"], data["y_train"])

    tmp_hist_model.best_model_state = None
    train_losses = get_losses(tmp_hist_model, data["x_train"], data["y_train"],
                              compute_weights=True)

    val_losses = get_losses(tmp_hist_model, data["x_val"], data["y_val"],
                            compute_weights=True)

    tmp_hist_model.best_iter = np.argmin(val_losses)
    tmp_hist_model.train_losses = train_losses
    tmp_hist_model.val_losses = val_losses

    return tmp_hist_model


def train_dnn_model(data, early_stopping=True, compute_val_weights=True,
                    max_iters=100):

    if "split_val" in data.keys():
        # for the DNN model, we can use early stopping internally,
        # but to use that we need to merge train and validation sets
        # first and then provide the respective split seed
        split_seed = data["split_val"]
        clsf_hist_model = PyTorchClassifier(
            layers=[64, 64, 64],
            validation_fraction=0.5,
            split_seed=split_seed,
        )
        x_train = np.concatenate((data["x_train"], data["x_val"]))
        y_train = np.concatenate((data["y_train"], data["y_val"]))
    else:
        clsf_hist_model = PyTorchClassifier(
            layers=[64, 64, 64],
            validation_fraction=None,
            epochs=100,
        )
        x_train = data["x_train"]
        y_train = data["y_train"]

    if compute_val_weights:
        weights = "balanced"
    else:
        weights = None

    steps = [('scaler', StandardScaler()), ('clsf', clsf_hist_model)]

    tmp_hist_model = HGBPipeline(steps)

    # Save seed for random split so train/val split can be reproduced
    if "split_val" in data.keys():
        tmp_hist_model.split_seed = data["split_val"]

    # balanced class weights will be computed internally
    tmp_hist_model.fit(x_train, y_train, clsf__sample_weights=weights)

    tmp_hist_model.val_losses = np.array(tmp_hist_model["clsf"].val_losses)
    tmp_hist_model.train_losses = np.array(tmp_hist_model["clsf"].train_losses)

    return tmp_hist_model


def train_hgb_model(data, early_stopping=True, compute_val_weights=True,
                    max_iters=100):

    # Compute class weights for training set for weighted loss
    train_sample_weights = get_sample_weights(data["y_train"])

    if compute_val_weights:
        sample_weights = get_sample_weights(data["y_val"])
    else:
        sample_weights = None

    clsf_hist_model = HistGradientBoostingClassifier(
        max_bins=127, class_weight="balanced", max_iter=1,
        early_stopping=False, warm_start=True)

    steps = [('scaler', StandardScaler()), ('clsf', clsf_hist_model)]

    tmp_hist_model = HGBPipeline(steps)

    # Save seed for random split so train/val split can be reproduced
    if "split_val" in data.keys():
        tmp_hist_model.split_seed = data["split_val"]

    min_val_loss = np.inf

    train_losses = []
    val_losses = []

    for i in range(max_iters):
        tmp_hist_model.fit(data["x_train"], data["y_train"])

        tmp_train_preds = tmp_hist_model.predict_proba(data["x_train"])[:, 1]
        tmp_train_loss = log_loss(data["y_train"], tmp_train_preds,
                                  sample_weight=train_sample_weights)

        tmp_val_preds = tmp_hist_model.predict_proba(
            data["x_val"]
            )[:, 1]

        tmp_val_loss = log_loss(data["y_val"], tmp_val_preds,
                                sample_weight=sample_weights)

        train_losses.append(tmp_train_loss)
        val_losses.append(tmp_val_loss)

        if tmp_val_loss < min_val_loss-1e-7:
            min_val_loss = tmp_val_loss
            iter_diff = 0
            tmp_hist_model.best_model_state = deepcopy(tmp_hist_model)
            tmp_hist_model.best_iter = i
        else:
            iter_diff += 1

        if early_stopping and (iter_diff >= 10):
            break

        tmp_hist_model["clsf"].max_iter += 1

    tmp_hist_model.train_losses = train_losses
    tmp_hist_model.val_losses = val_losses

    return tmp_hist_model


def train_model_ensemble(data, num_models=10, cv_mode="fixed",
                         max_iters=100, model_type="HGB",
                         compute_val_weights=True,
                         save_model_dir=None,
                         early_stopping=True):
    """
    Trains an ensemble of tree-based models and returns the
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
        max_iters (int, optional): The maximum number of iterations to train
            each model for. Defaults to 100.
        model_type (str, optional): The type of model to train. Currently,
            "HGB" (HistGradientBoostingClassifier),
            "RF" (RandomForestClassifier) and "Ada" (AdaboostClassifier)
            are supported. Default is "HGB".
        compute_val_weights (bool, optional): Whether to compute weights for
            the validation set. Defaults to True.
        save_model_dir (str, optional): The directory to save the trained
            models to. If None, the models will not be saved. Default is None.
        early_stopping (bool, optional): Whether or not to use early stopping
            during training. Default is True.

    Returns:
        Tuple containing the following elements:
        - ens_mean_preds (array-like): The mean predictions of the HGB ensemble
            on the test set, with shape (x_test.shape[0],).
        - loss_dict (dict): A dictionary containing the training, validation
            and test losses for each model in the ensemble.
        - model_list (list): A list containing the trained HGB models.
    """

    assert cv_mode in ["fixed", "random", "k-fold"], (
        "cv_mode must be either 'fixed', 'random' or 'k-fold'"
        )

    model_list = []

    loss_dict = {}
    cv_data = generate_cv_data(data, num_models, cv_mode)
    for ens, dat in zip(range(num_models), cv_data):

        if model_type == "HGB":
            tmp_hist_model = train_hgb_model(
                dat, early_stopping=early_stopping,
                compute_val_weights=compute_val_weights,
                max_iters=max_iters)

            tmp_hist_model.cv_mode = cv_mode
        elif model_type == "RF":
            tmp_hist_model = train_rf_model(
                dat,
                max_iters=max_iters)

            tmp_hist_model.cv_mode = cv_mode
        elif model_type == "Ada":
            tmp_hist_model = train_ada_model(
                dat, early_stopping=early_stopping,
                max_iters=max_iters)

            tmp_hist_model.cv_mode = cv_mode
        elif model_type == "DNN":
            tmp_hist_model = train_dnn_model(
                dat, early_stopping=early_stopping,
                max_iters=max_iters)

            tmp_hist_model.cv_mode = cv_mode
        else:
            raise NotImplementedError

        if save_model_dir is not None:
            if model_type == "DNN":
                save_dnn_model(tmp_hist_model, save_model_dir, ens)
            else:
                save_model(tmp_hist_model, save_model_dir, ens)

        model_list.append(tmp_hist_model)

        tmp_val_losses = tmp_hist_model.val_losses

        tmp_train_losses = tmp_hist_model.train_losses

        # For each model in the ensemble, put losses in dictionary to return
        loss_dict[f"model_{ens}"] = {
                "train_loss": tmp_train_losses,
                "val_loss": tmp_val_losses,
            }

    return loss_dict, model_list


def eval_single_HGB_model(model, data):
    test_preds = model.best_model_state.predict_proba(data["x_test"])[:, 1]
    return test_preds


def eval_single_dnn_model(model, data):
    return model.predict_proba(data["x_test"])


def eval_single_adaboost_model(model, data):
    preds = model.staged_predict_proba(data["x_test"])
    for idx, pred in enumerate(preds):
        if idx == model.best_iter:
            return pred[:, 1]


def eval_single_rf_model(model, data):
    test_preds = model.predict_proba(data["x_test"])[:, 1]

    return test_preds


def eval_single_model(model, data, model_type="HGB"):
    """Evaluate a single model on the test set.

    Args:
        model: A trained single trained model.
        data (dict): A dictionary containing the test set features.
        model_type (str, optional): The type of model to evaluate. Currently,
            "HGB" (HistGradientBoostingClassifier),
            "RF" (RandomForestClassifier) and "Ada" (AdaboostClassifier)
            are supported. Default is "HGB".

    Returns:
        An array of predictions for the test set.
    """

    if model_type == "HGB":
        return eval_single_HGB_model(model, data)
    elif model_type == "RF":
        return eval_single_rf_model(model, data)
    elif model_type == "Ada":
        return eval_single_adaboost_model(model, data)
    elif model_type == "DNN":
        return eval_single_dnn_model(model, data)
    else:
        raise NotImplementedError


def eval_ensemble(all_models, data, model_type="HGB"):
    """Evaluate an ensemble of models on the test set.

    Args:
        all_models: A list of lists, where each sublist contains the
            trained HGB models for one run.
        data (dict): A dictionary containing the training, validation and test
            sets as well as the corresponding labels.

    Returns:
        A numpy array of shape (num_runs, x_test.shape[0]) containing the mean
        predictions of each HGB ensemble on the test set.
    """
    for run in range(len(all_models)):
        for idx, model in enumerate(all_models[run]):

            test_preds = eval_single_model(
                model, data,
                model_type=model_type)
            if idx == 0:
                ens_preds = test_preds
            else:
                ens_preds = np.vstack([ens_preds, test_preds])

        current_preds = np.mean(ens_preds, axis=0).reshape((1, -1))
        if run == 0:
            all_preds = current_preds
        else:
            all_preds = np.vstack([all_preds, current_preds])

    return all_preds


def train_model_multirun(data,
                         num_runs=10, ensembles_per_model=10,
                         cv_mode="fixed", max_iters=100,
                         model_type="HGB",
                         compute_val_weights=True,
                         save_model_dir=None,
                         early_stopping=True):
    """
    Run multible ensembles of HGB trainings and return array of mean test
    predictions for each ensemble.

    Args:
        data (dict): A dictionary containing the training, validation and test
            sets as well as the corresponding labels.
        num_runs (int, optional): The number of HGB ensemble trainings to run.
            Default is 10.
        ensembles_per_model (int, optional): The number of ensembles to train
            per HGB ensemble. Default is 10.
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
        max_iters (int, optional): The maximum number of iterations to run each
            HGB training for. Default is 100.
        model_type (str, optional): The type of model to train. Currently, only
            "HGB" (HistGradientBoostingClassifier) is supported.
            Defaults to "HGB".
        compute_val_weights (bool, optional): Whether or not to compute
            validation weights during training. Default is True.
        save_model_dir (str, optional): The directory to save the trained
            models to. If None, the models will not be saved. Default is None.
        early_stopping (bool, optional): Whether or not to use early stopping
            during training. Default is True.

    Returns:
        Tuple containing the following elements:
        - full_losses (dict): A dictionary containing the training, validation
            and test losses for each model in the ensemble for all runs.
        - all_models (list): A list of lists, where each sublist contains the
            trained HGB models for one run.

    """
    if cv_mode not in ["fixed", "random", "k-fold"]:
        raise ValueError(
            "cv_mode must be either 'fixed', 'random' or 'k-fold'"
            )

    all_models = []
    full_losses = {}
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")

        if save_model_dir is not None:
            save_model_dir_run = join(save_model_dir, f"run_{run}")
            if not exists(save_model_dir_run):
                makedirs(save_model_dir_run)
        else:
            save_model_dir_run = None

        losses, models = train_model_ensemble(
            data,
            num_models=ensembles_per_model, cv_mode=cv_mode,
            max_iters=max_iters, model_type=model_type,
            compute_val_weights=compute_val_weights,
            save_model_dir=save_model_dir_run,
            early_stopping=early_stopping
            )

        all_models.append(models)
        full_losses[f"run_{run}"] = losses

    return full_losses, all_models


def loss_ndarray_from_dict(loss_dict):
    """Convert a loss dictionary to a 2D numpy array.

    Args:
        loss_dict (dict): A dictionary containing the losses for each model in
            the ensemble for all runs.

    Returns:
        ndarray: A 3D numpy array containing the losses for each model in the
        ensemble for all runs.

    """
    num_runs = len(loss_dict.keys())
    ensembles_per_model = len(loss_dict["run_0"].keys())
    num_iters = len(loss_dict["run_0"]["model_0"]["train_loss"])
    train_loss_arr = np.zeros((num_runs, ensembles_per_model, num_iters))
    val_loss_arr = np.zeros((num_runs, ensembles_per_model, num_iters))

    for i in range(num_runs):
        for j in range(ensembles_per_model):
            train_loss_arr[i, j, :] = loss_dict[f"run_{i}"][f"model_{j}"][
                "train_loss"]
            val_loss_arr[i, j, :] = loss_dict[f"run_{i}"][f"model_{j}"][
                "val_loss"]

    return {"train_loss": train_loss_arr, "val_loss": val_loss_arr}


def generate_cv_data(data, num_models, cv_mode="fixed"):
    """Generate cross-validation data for training an ensemble.

    Args:
        data (dict): A dictionary containing the training, validation and test
            sets as well as the corresponding labels.
        num_models (int): The number of models in the ensemble.
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
                `num_models` models each trained on a different
                train/validation k-fold assignment)
            Defaults to "fixed".

    Yields:
        dict: A dictionary containing the training, validation and test sets
            as well as the corresponding labels for each model in the ensemble.
    """
    if cv_mode == "k-fold":
        x_full = np.vstack([data["x_train"], data["x_val"]])
        y_full = np.hstack([data["y_train_databg"], data["y_val_databg"]])
        x_split = np.array_split(x_full, num_models)
        y_split = np.array_split(y_full, num_models)

    cv_data = {}

    for i in range(num_models):
        if cv_mode == "random":
            cv_data["split_val"] = np.random.randint(0, 100000)
            x_full = np.vstack([data["x_train"], data["x_val"]])
            y_full = np.hstack([data["y_train_databg"], data["y_val_databg"]])

            (cv_data["x_train"], cv_data["x_val"],
             cv_data["y_train"], cv_data["y_val"]) = train_test_split(
                x_full, y_full, test_size=0.5,
                random_state=cv_data["split_val"]
                )

            cv_data["x_test"] = data["x_test"]

        elif cv_mode == "k-fold":
            cv_data["x_val"] = x_split[i]
            cv_data["y_val"] = y_split[i]
            cv_data["x_train"] = np.concatenate(
                [x_split[j] for j in range(num_models) if j != i]
                )

            cv_data["y_train"] = np.concatenate(
                [y_split[j] for j in range(num_models) if j != i]
                )

            cv_data["x_test"] = data["x_test"]

        else:
            cv_data["x_train"] = data["x_train"]
            cv_data["y_train"] = data["y_train_databg"]
            cv_data["x_val"] = data["x_val"]
            cv_data["y_val"] = data["y_val_databg"]
            cv_data["x_test"] = data["x_test"]

        yield cv_data
