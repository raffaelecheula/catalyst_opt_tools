# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# PLOT CUMULATIVE MAX CURVE
# -------------------------------------------------------------------------------------

def plot_cumulative_max_curve(
    data_all: list,
    n_runs: int = None,
    key_y: str = "rate",
    key_run: str = "run",
    ax: object = None,
    x_max: float = None,
    y_max: float = None,
    color: str = "crimson",
    alpha_fill: float = 0.2,
    xlabel: str = "Number of structures evaluated [-]",
    ylabel: str = "Maximum reaction rate [1/s]",
    filename: str = None,
    plot_mean: bool = False,
):
    """
    For each run, calculate the maximum y value reached as a function of the steps
    (evaluations). Then, calculate the mean value between the runs.
    """
    # Initialize axis object if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    # Initialize a list to store the maximum y values for each run.
    yy_max_all = []
    n_runs = n_runs if n_runs else max([data[key_run] for data in data_all])+1
    for run in range(n_runs):
        # Extract y values for the current run.
        yy_list = [data[key_y] for data in data_all if data[key_run] == run]
        # Calculate maximum y values for the current run.
        yy_max_list = [0.]
        for ii, yy in enumerate(yy_list):
            yy_max = yy if len(yy_max_list) == 1 else max(yy_max, yy)
            yy_max_list.append(yy_max)
        # Append the maximum y values for the current run to the list.
        yy_max_all.append(yy_max_list)
        # Plot maximum y values for the current run.
        if plot_mean is False:
            ax.plot(yy_max_list, linestyle="--", color=color)
    # Calculate mean, min, and max of the maximum y values across all runs.
    yy_max_mean = np.mean(yy_max_all, axis=0)
    yy_max_min = np.min(yy_max_all, axis=0)
    yy_max_max = np.max(yy_max_all, axis=0)
    if plot_mean is True:
        # Plot the mean curve.
        ax.plot(yy_max_mean, color=color)
        # Fill between min and max.
        x_axis = np.arange(len(yy_max_mean))
        ax.fill_between(x_axis, yy_max_min, yy_max_max, color=color, alpha=alpha_fill)
    # Get x and y max for the plot.
    x_max = x_max if x_max else len(yy_max_mean) - 1
    y_max = y_max if y_max else max([data[key_y] for data in data_all]) * 1.10
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    # Set axes labels.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Save the plot if filename is provided.
    if filename is not None:
        plt.savefig(filename)
    # Return the axis object.
    return ax

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------