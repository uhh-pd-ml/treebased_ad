import numpy as np
from matplotlib import pyplot as plt
from utils import eval_ensemble, multi_roc_sigeffs


def get_sic_curves_multirun(ax, multi_tprs, multi_fprs, y_test,
                            max_rel_err=0.2, label="", color=None,
                            linestyle=None):
    """Plot single SIC curve including errror bands for several runs.

    Errorbands are defined as inner 68% of SIC value distribution of the
    independent runs

    Args:
        ax (matplotlib.pyplot.axis): Matplotlib axis object to plot on.
        multi_tprs (numpy.ndarray): Array of shape
            (num_runs, num_tpr_values) containing the TPR values for each run
        multi_fprs (numpy.ndarray): Array of shape
            (num_runs, num_fpr_values) containing the FPR values for each run
        y_test (numpy.ndarray): Truth labels of test set.
        max_rel_err (float, optional): Maximum relative error allowed
            w.r.t. number of background events up to which we still plot the
            SIC curve.
        label (str, optional): Label for the plot legend.

    Returns:
        matplotlib.pyplot.axis: Matplotlib axis object containing plot of
            median SIC with error bands.

    """
    multi_sics = multi_tprs/np.sqrt(multi_fprs)

    median_tprs = np.median(multi_tprs, axis=0)
    median_fprs = np.median(multi_fprs, axis=0)

    median_sics = np.median(multi_sics, axis=0)
    upper_sics = np.percentile(multi_sics, 84, axis=0)
    lower_sics = np.percentile(multi_sics, 16, axis=0)

    plot_mask = (
        (1/np.sqrt(median_fprs*y_test[y_test == 0].shape[0])) < max_rel_err
        )

    if linestyle is None:
        linestyle = 'solid'
    else:
        assert linestyle in ['solid', 'dashed', 'dashdot', 'dotted'], (
            "Error! `linestyle` must be one of 'solid', 'dashed', 'dashdot' "
            "or 'dotted'."
            )

    if color is None:
        p = ax.plot(median_tprs[plot_mask], median_sics[plot_mask],
                    label=label, linestyle=linestyle)
    else:
        p = ax.plot(median_tprs[plot_mask], median_sics[plot_mask],
                    label=label, color=color, linestyle=linestyle)

    ax.fill_between(median_tprs[plot_mask], upper_sics[plot_mask],
                    lower_sics[plot_mask], alpha=0.2,
                    facecolor=p[0].get_color(),
                    edgecolor=None)

    return ax


def plot_sic_curves(tpr_list, fpr_list, y_test_list, max_rel_err=0.2,
                    color_list=None, title=None, linestyles=None,
                    xlabel="TPR", ylabel="SIC", out_filename=None, labels=None,
                    legend_loc="upper right", max_y=None):
    """Plot single SIC curve including errror bands for several runs.

    Args:
        tpr_list (list of numpy.ndarray): List of arrays of shape
            (num_tpr_values) containing the TPR values for each run
        fpr_list (list of numpy.ndarray): List of arrays of shape
            (num_fpr_values) containing the FPR values for each run
        y_test_list (list of numpy.ndarray): List of arrays containing the
            truth labels of the test set for each study that should be plotted.
        max_rel_err (float, optional): Maximum relative error allowed
            w.r.t. number of background events up to which we still plot the
            SIC curve.
        xlabel (str, optional): String containing x axis label.
            Default is 'TPR'.
        ylabel (str, optional): String containing y axis label.
            Default is 'SIC'.
        out_filename (str, optional): String describing the filename under
            which the plot should be saved.
        labels (NoneType or list of str, optional):
            List of labels describing the different training runs that should
            be plotted.
        legend_loc (str, optional): String describing the location of the
            legend. Default is 'upper right'.
        max_y (NoneType or float, optional): Maximum value of y axis.
            Default is None.
    """

    f, ax = plt.subplots()

    if (labels is not None) and (len(labels) != len(tpr_list)):
        raise ValueError(("Error! `labels` must have same length as "
                          "`tpr_val_list` and `fpr_val_list`"))

    if (color_list is not None) and (len(color_list) != len(tpr_list)):
        raise ValueError(("Error! `color_list` must have same length as "
                          "`tpr_val_list` and `fpr_val_list`"))

    if (linestyles is not None) and (len(linestyles) != len(tpr_list)):
        raise ValueError(("Error! `color_list` must have same length as "
                          "`tpr_val_list` and `fpr_val_list`"))

    if labels is None:
        labels = [None]*len(tpr_list)

    if color_list is None:
        color_list = [None]*len(tpr_list)

    if linestyles is None:
        linestyles = [None]*len(tpr_list)

    for i in range(len(tpr_list)):
        get_sic_curves_multirun(ax, tpr_list[i], fpr_list[i],
                                y_test_list[i], color=color_list[i],
                                linestyle=linestyles[i],
                                max_rel_err=max_rel_err, label=labels[i])

    plt.xlabel(xlabel)
    plt.xlim(0, 1)
    plt.ylim(0, max_y)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc, frameon=False)
    if title is not None:
        ymin = 0
        ymax = max_y
        xmin = 0
        xmax = 1
        a = 0.03
        plt.text(xmin + a * (xmax-xmin), ymin + (1-a) * (ymax-ymin),
                 title, size=plt.rcParams['axes.labelsize'],
                 color='black', horizontalalignment='left',
                 verticalalignment='top')

    if out_filename is not None:
        plt.savefig(out_filename)
    plt.show()
    plt.close()


def plot_sic_curve_comparison(model_list, data, out_filename=None,
                              model_types=None,
                              labels=None, xlabel="TPR", ylabel="SIC",
                              legend_loc="upper right", max_rel_err=0.2,
                              max_y=None):
    """A simple wrapper function of `get_sic_curves_multirun`

    This function allows to plot multiple SIC curves (e.g. when comparing
    several runs with different training settings) including their
    respective error bands on one plot.

    Args:
        model_list (list of list): List of models for which the SIC curves
            should be plotted. The structure of the list should be
            as follows: The first index refers to a separate "study" (e.g.
            using vanilla input settings or 10 gaussian noise features added).
            The second index is a single run of this study. The third index
            refers to the specific model within the ensemble for that run.
        data (dict or list of dict): Dictionary or list of dictionaries
            (one for each study) containing the data to be used for evaluation.
            Should at least contain the keys "x_test" and "y_test".
        out_filename (str, optionaö): String describing the filename under
            which the plot should be saved.
        model_types (NoneType or list of str, optional): List of strings
            describing the model types of the models in `model_list`.
        labels (NoneType or list of str, optional):
            List of labels describing the different training runs that should
            be plotted for comparison. The length of the list *must* be equal
            to the number of runs or a `ValueError` is raised.
        xlabel (str, optional): String containing x axis label.
            Default is 'TPR'.
        ylabel (str, optional): String containing y axis label.
            Default is 'SIC'.
        legend_loc (str, optional): Location parameter of
            `matplotlib.pyplot.legend`. Default is "upper right".
        max_rel_err (float, optional): Maximum relative error allowed
            w.r.t. number of background events up to which we still plot the
            SIC curve.
        show (bool, optional): Boolean deciding whether plots should
            also be directly shown instead of just being stored to disk.
            Default is True.
        max_y (NoneType or float, optional): Numeric value
            defining the maximum y value up to which the SIC is plotted.
            Default is None, leading to `matplotlib` figuring out the limit
            by itself.
    """

    if labels is not None:
        assert len(model_list) == len(labels), (
            "Error! `labels` must have same length as `model_list`"
            )

    if model_types is not None:
        assert len(model_list) == len(model_types), (
            "Error! `model_types` must have same length as `model_list`"
            )
    else:
        print(("No model types provided. Assuming all models are scikit-learn "
               "HistGradientBoostingClassifier instances."))
        model_types = ["HGB"]*len(model_list)

    if type(data) is not list:
        data = [data]*len(model_list)
    else:
        assert len(model_list) == len(data), (
            "Error! List of `data` must have same length as `model_list`"
            )

    tpr_val_list = []
    fpr_val_list = []
    for i in range(len(model_list)):

        if model_types is None:
            model_types = ["HGB"]*len(model_list[i])

        full_preds_tmp = eval_ensemble(
            model_list[i], data[i],
            model_type=model_types[i],
            )

        tpr_vals_tmp, fpr_vals_tmp = multi_roc_sigeffs(full_preds_tmp,
                                                       data[i]["y_test"])
        tpr_val_list.append(tpr_vals_tmp)
        fpr_val_list.append(fpr_vals_tmp)

    y_test_list = [data[i]["y_test"] for i in range(len(data))]

    plot_sic_curves(tpr_val_list, fpr_val_list, y_test_list,
                    max_rel_err=max_rel_err, xlabel=xlabel, ylabel=ylabel,
                    out_filename=out_filename, labels=labels,
                    legend_loc=legend_loc, max_y=max_y)


def plot_losses(losses_to_plot, out_file="./losses.pdf", labels=None,
                show=True, xlims=None, ylims=None):
    """
    Plot the given list of loss values as a line graph and save the figure as
    a PDF file.

    Args:
        losses_to_plot (list or array): A list of arrays or lists, where each
            element represents a particular set of loss values (e.g. for
            different models or training vs validation losses etc.).
        out_file (str): The path to save the output PDF file.
            Default is "./losses.pdf".
        labels (list, optional): A list of strings, where each element
            represents the label for a particular model's loss values.
            If None (default), no labels will be shown.
        show (bool): A boolean indicating whether to show the plot in a window.
            Default is True.
        xlims (tuple, optional): A tuple of two values representing the lower
            and upper limits of the x-axis.
        ylims (tuple, optional): A tuple of two values representing the lower
            and upper limits of the y-axis.

    Raises:
        ValueError: If the lengths of the labels parameter and input
        loss list/array do not match.

    """

    if labels is not None and len(losses_to_plot) != len(labels):
        raise ValueError(
            "Lengths of labels parameter and input loss list/array must match!"
            )

    if labels is None:
        labels = [None]*len(losses_to_plot)

    for i, loss in enumerate(losses_to_plot):
        xvals = np.arange(1, len(loss)+1)
        plt.plot(xvals, loss, label=loss[i])

    plt.xlabel("Iteration")
    plt.ylabel("val loss")
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.legend(loc="upper right")
    plt.savefig(out_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
