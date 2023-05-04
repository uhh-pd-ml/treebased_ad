import numpy as np
from matplotlib import pyplot as plt


def get_sic_curves_multirun(ax, multi_tprs, multi_fprs, y_test,
                            max_rel_err=0.2, label=""):
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

    p = ax.plot(median_tprs[plot_mask], median_sics[plot_mask], label=label)
    ax.fill_between(median_tprs[plot_mask], upper_sics[plot_mask],
                    lower_sics[plot_mask], alpha=0.2, color=p[0].get_color())

    return ax


def plot_sic_curve_comparison(tpr_val_list, fpr_val_list, y_test, out_filename,
                              labels=None, xlabel="TPR", ylabel="SIC",
                              legend_loc="upper right", max_rel_err=0.2,
                              show=True, max_y=None):
    """A simple wrapper function of `get_sic_curves_multirun`

    This function allows to plot multiple SIC curves (e.g. when comparing
    several runs with different training settings) including their
    respective error bands on one plot.

    Args:
        tpr_val_list (numpy.ndarray): Numpy array of shape (runs, thresholds),
            containing true positive rate (TPR) values for each chosen
            threshold cut of the respective model predictions.
        fpr_val_list (numpy.ndarray): Same as `tpr_val_list`, but with false
            positive rate (FPR) values instead.
        y_test (numpy.ndarray): Numpy array containing the labels of the test
            set (should be 0 for background and 1 for signal events).
        out_filename (str): String describing the filename under which the
            plot should be saved.
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

    Raises:
        ValueError: If `labels` is provided, but its length is not equal to
            the length of `tpr_val_list`.

    """

    f, ax = plt.subplots()

    if (labels is not None) and (len(labels) != len(tpr_val_list)):
        raise ValueError(("Error! `labels` must have same length as "
                          "`tpr_val_list` and `fpr_val_list`"))

    if labels is None:
        labels = [None]*len(tpr_val_list)

    for i in range(len(tpr_val_list)):
        get_sic_curves_multirun(ax, tpr_val_list[i], fpr_val_list[i], y_test,
                                max_rel_err=max_rel_err, label=labels[i])

    plt.xlabel(xlabel)
    plt.xlim(0, 1)
    plt.ylim(0, max_y)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.savefig(out_filename)
    if show:
        plt.show()
    plt.close()


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
