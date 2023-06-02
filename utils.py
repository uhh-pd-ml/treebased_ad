import numpy as np
from os.path import join, exists, isdir
from os import makedirs, listdir
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


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


def load_models_allruns(model_dir):
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
            models_per_run = len(listdir(join(model_dir, f"run_{i}")))

        tmp_model_list = []
        for j in range(models_per_run):
            tmp_model_list.append(load_single_model(model_dir, i, j))

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


def load_lhco_rd(data_dir):
    """Load the LHCO R&D dataset.

    This function loads the LHCO R&D dataset from a specified directory.
    The dataset consists of 3 sets: training set, validation set and test set.
    For the training and validation set, the data/background labels are
    needed, while for the test set, signal/background labels are required. The
    function performs the following steps:

        1. Loads the training set, validation set and testset including their
           respective "extra" background samples.
        2. Concatenates original and extra background samples for each dataset.
        3. Shuffles the training and validation sets.
        5. Converts the data to the float32 data type (needed for
           pytorch DNN classifier training).

    Args:
        data_dir (str): The path to the directory containing the LHCO dataset.

    Returns:
        X_train (array-like, shape=(n_samples_train, n_features)):
            The loaded training set data.
        y_train (array-like, shape=(n_samples_train,)):
            The loaded training set labels.
        X_val (array-like, shape=(n_samples_val, n_features)):
            The loaded validation set data.
        y_val (array-like, shape=(n_samples_val,)):
            The loaded validation set labels.
        X_test (array-like, shape=(n_samples_test, n_features)):
            The loaded test set data.
        y_test (array-like, shape=(n_samples_test,)):
            The loaded test set labels.

    """

    # for train and val set, we only need data/bg labels
    X_train = np.load(join(data_dir, "innerdata_train.npy"))[:, 1:-1]
    X_train_extrabg = np.load(
        join(data_dir, "innerdata_extrabkg_train.npy")
        )[:, 1:-1]

    y_train = np.concatenate((np.ones((X_train.shape[0], )),
                              np.zeros((X_train_extrabg.shape[0], ))))

    X_train = np.concatenate((X_train, X_train_extrabg))

    # shuffle train set
    shuffle_arr = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_arr)
    X_train = X_train[shuffle_arr]
    y_train = y_train[shuffle_arr]

    X_val = np.load(join(data_dir, "innerdata_val.npy"))[:, 1:-1]
    X_val_extrabg = np.load(
        join(data_dir, "innerdata_extrabkg_val.npy")
        )[:, 1:-1]

    y_val = np.concatenate((np.ones((X_val.shape[0], )),
                            np.zeros((X_val_extrabg.shape[0], ))))

    X_val = np.concatenate((X_val, X_val_extrabg))

    # shuffle val set
    shuffle_arr_val = np.arange(X_val.shape[0])
    np.random.shuffle(shuffle_arr_val)
    X_val = X_val[shuffle_arr_val]
    y_val = y_val[shuffle_arr_val]

    # for test set, we only need sig/bg labels
    X_test = np.load(join(data_dir, "innerdata_test.npy"))[:, 1:-1]
    y_test = np.load(join(data_dir, "innerdata_test.npy"))[:, -1]
    X_test_extrabg = np.load(
        join(data_dir, "innerdata_extrabkg_test.npy")
        )[:, 1:-1]

    y_test_extrabg = np.zeros((X_test_extrabg.shape[0], ))
    X_test = np.concatenate((X_test, X_test_extrabg))
    y_test = np.concatenate((y_test, y_test_extrabg))

    X_train = X_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def add_gaussian_features(x_train, x_val, x_test, n_gaussians=10):
    """Adds Gaussian-distributed random variables to input features

    Args:
        x_train (numpy.ndarray): Training set features.
        x_val (numpy.ndarray): Validation set features.
        x_test (numpy.ndarray): Test set features.
        n_gaussians (int, optional): Number of Gaussian random variables to
            generate and add to the input data. Default is 10.

    Returns:
        tuple: Tuple containing the updated features for the training,
        validation and test sets, respectively, with Gaussian-distributed
        random variables added as additional features.
    """
    rand_var_train = np.random.randn(x_train.shape[0], n_gaussians)
    rand_var_val = np.random.randn(x_val.shape[0], n_gaussians)
    rand_var_test = np.random.randn(x_test.shape[0], n_gaussians)

    x_train_gaus = np.hstack([x_train, rand_var_train]).astype(np.float32)
    x_val_gaus = np.hstack([x_val, rand_var_val]).astype(np.float32)
    x_test_gaus = np.hstack([x_test, rand_var_test]).astype(np.float32)

    return x_train_gaus, x_val_gaus, x_test_gaus


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
        hist_model: a trained scikit-learn's HistGradientBoostingClassifier
            model.
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


def train_histgradboost_ensemble(x_train, y_train, x_val, y_val, x_test,
                                 y_test, num_models=10, cv_mode="fixed",
                                 max_iters=100, compute_val_weights=True,
                                 save_full_preds=None, save_model_dir=None):
    """
    Trains an ensemble of HistGradientBoostingClassifier models and returns the
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
        max_iters (int, optional): The maximum number of iterations to train
            each model for. Defaults to 100.
        compute_val_weights (bool, optional): Whether to compute weights for
            the validation set. Defaults to True.
        save_full_preds (str, optional): The filename to save the full
            predictions of the ensemble on the test set as a .npy file.
            Defaults to None (in which case no predictions will be saved).

    Returns:
        tuple: A tuple containing the mean predictions on the test set, the
            training losses for each model in the ensemble, the validation
            losses for each model in the ensemble, the test losses for each
            model in the ensemble, and a list of the trained
            HistGradientBoostingClassifier models in the ensemble.

    """

    if cv_mode == "k-fold":

        x_full = np.vstack([x_train, x_val])
        y_full = np.hstack([y_train, y_val])
        x_split = np.array_split(x_full, num_models)
        y_split = np.array_split(y_full, num_models)

    model_list = []

    for ens in range(num_models):
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

        # Train HGB model on training set. Don't use any early stopping (will
        # train maximum of 100 iterations, which can be changed using
        # max_iter parameter -> sklearn default)
        clsf_hist_model = HistGradientBoostingClassifier(
            max_bins=127, class_weight="balanced", max_iter=max_iters,
            early_stopping=False)

        steps = [('scaler', StandardScaler()), ('HGB', clsf_hist_model)]

        tmp_hist_model = HGBPipeline(steps)
        tmp_hist_model.fit(x_train_tmp, y_train_tmp)
        if save_model_dir is not None:
            save_model(tmp_hist_model, save_model_dir, ens)

        model_list.append(tmp_hist_model)

        # Find out optimal iteration for this model on validation set,
        # then get predictions of this model on test set
        tmp_model_preds = preds_from_optimal_iter(
            tmp_hist_model, x_val_tmp, y_val_tmp, x_test_tmp,
            compute_weights=compute_val_weights)

        tmp_model_preds = tmp_model_preds.reshape((1, -1))

        tmp_val_losses = get_losses(tmp_hist_model, x_val_tmp, y_val_tmp,
                                    compute_weights=compute_val_weights)

        tmp_val_losses = tmp_val_losses.reshape((1, -1))

        tmp_test_losses = get_losses(tmp_hist_model, x_test_tmp, y_test,
                                     compute_weights=True)

        tmp_test_losses = tmp_test_losses.reshape((1, -1))

        tmp_train_losses = get_losses(tmp_hist_model, x_train_tmp, y_train_tmp)
        tmp_train_losses = tmp_train_losses.reshape((1, -1))

        # For each model in the ensemble, stack the test predictions
        if ens == 0:
            ens_preds = tmp_model_preds
            ens_val_losses = tmp_val_losses
            ens_train_losses = tmp_train_losses
            ens_test_losses = tmp_test_losses
        else:
            ens_preds = np.vstack([ens_preds, tmp_model_preds])
            ens_val_losses = np.vstack([ens_val_losses, tmp_val_losses])
            ens_train_losses = np.vstack([ens_train_losses, tmp_train_losses])
            ens_test_losses = np.vstack([ens_test_losses, tmp_test_losses])

    if save_full_preds is not None:
        print(f"Saving full predictions as {save_full_preds}")
        np.save(save_full_preds, ens_preds)

    # Finally, take mean of all predictions in the ensemble
    ens_mean_preds = np.mean(ens_preds, axis=0)

    return (ens_mean_preds, ens_train_losses,
            ens_val_losses, ens_test_losses, model_list)


def train_histgradboost_multi(x_train, y_train, x_val, y_val, x_test, y_test,
                              num_runs=10, ensembles_per_model=10,
                              cv_mode="fixed", max_iters=100,
                              compute_val_weights=True,
                              save_ensemble_preds=False, save_model_dir=None):
    """
    Run multible ensembles of HGB trainings and return array of mean test
    predictions for each ensemble.

    Args:
        x_train (array-like): The training data to fit the HGB model.
        y_train (array-like): The target values for the training data.
        x_val (array-like): The validation data to fit the HGB model.
        y_val (array-like): The target values for the validation data.
        x_test (array-like): The test data to predict using the trained HGB
            models.
        y_test (array-like): The true labels for the test data.
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
        compute_val_weights (bool, optional): Whether or not to compute
            validation weights during training. Default is True.
        save_ensemble_preds (bool, optional): Whether or not to save the full
            ensemble predictions during training. Default is False.

    Returns:
        Tuple containing the following elements:
        - full_preds (array-like): The mean predictions of each HGB ensemble on
          the test set, with shape (num_runs, x_test.shape[0]).
        - full_train_losses (array-like): The training losses of each HGB
          ensemble, with shape (num_runs, ensembles_per_model, max_iters).
        - full_val_losses (array-like): The validation losses of each HGB
          ensemble, with shape (num_runs, ensembles_per_model, max_iters).
        - full_test_losses (array-like): The test losses of each HGB ensemble,
          with shape (num_runs, ensembles_per_model, max_iters).
        - all_models (list): A list of lists, where each sublist contains the
          trained HGB models for one run.

    """
    if cv_mode not in ["fixed", "random", "k-fold"]:
        raise ValueError(
            "cv_mode must be either 'fixed', 'random' or 'k-fold'"
            )

    all_models = []
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        if save_ensemble_preds:
            save_str = f"./ensemble_preds_run{run}.npy"
        else:
            save_str = None

        if save_model_dir is not None:
            save_model_dir_run = join(save_model_dir, f"run_{run}")
            if not exists(save_model_dir_run):
                makedirs(save_model_dir_run)

        (ens_mean_preds, ens_train_losses, ens_val_losses,
         ens_test_losses, models) = train_histgradboost_ensemble(
            x_train, y_train, x_val, y_val, x_test, y_test,
            num_models=ensembles_per_model, cv_mode=cv_mode,
            max_iters=max_iters, compute_val_weights=compute_val_weights,
            save_full_preds=save_str, save_model_dir=save_model_dir_run
            )

        all_models.append(models)
        ens_mean_preds = ens_mean_preds.reshape((1, -1))
        ens_val_losses = ens_val_losses.reshape((1, ensembles_per_model, -1))
        ens_train_losses = ens_train_losses.reshape(
            (1, ensembles_per_model, -1)
            )

        ens_test_losses = ens_test_losses.reshape((1, ensembles_per_model, -1))

        if run == 0:
            full_preds = ens_mean_preds
            full_train_losses = ens_train_losses
            full_val_losses = ens_val_losses
            full_test_losses = ens_test_losses
        else:
            full_preds = np.vstack([full_preds, ens_mean_preds])
            full_train_losses = np.vstack(
                [full_train_losses, ens_train_losses]
                )
            full_val_losses = np.vstack([full_val_losses, ens_val_losses])
            full_test_losses = np.vstack([full_test_losses, ens_test_losses])

    return (full_preds, full_train_losses, full_val_losses,
            full_test_losses, all_models)
