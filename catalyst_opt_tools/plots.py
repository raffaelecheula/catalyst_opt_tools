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
    n_x_ticks: int = 10,
    xlabel: str = "Number of structures evaluated [-]",
    ylabel: str = "Maximum reaction rate [1/s]",
    filename: str = None,
    plot_mean: bool = False,
) -> object:
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
    n_runs = n_runs if n_runs else max([data[key_run] for data in data_all]) + 1
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
    # Set the number of x ticks.
    ax.locator_params(axis="x", nbins=n_x_ticks)
    # Set axes labels.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Save the plot if filename is provided.
    if filename is not None:
        plt.savefig(filename)
    # Return the axis object.
    return ax

# -------------------------------------------------------------------------------------
# PLOT HALF VIOLINS
# -------------------------------------------------------------------------------------        

def plot_half_violins(
    data_all: list,
    n_runs: int = None,
    key_y: str = "rate",
    key_run: str = "run",
    ax: object = None,
    x_max: float = None,
    y_max: float = None,
    n_violins: int = 10,
    color: str = "crimson",
    edgecolor: str = "black",
    alpha_fill: float = 0.7,
    n_x_ticks: int = 10,
    merge_data_all: bool = True,
    scatter_top_n: int = None,
    xlabel: str = "Number of structures evaluated [-]",
    ylabel: str = "Reaction rate distrbution [1/s]",
    filename: str = None,
    kwargs_violin: dict = {"showmeans": False, "showextrema": False, "points": 300},
    kwargs_scatter: dict = {"color": "black", "facecolors": "none", "s": 10},
):
    """
    Plot distributions of y values as half violins at different steps.
    """
    # Initialize axis object if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    # Determine number of steps and size of each group.
    n_steps = len([data[key_y] for data in data_all if data[key_run] == 0])
    if n_steps % n_violins != 0:
        raise ValueError("n_violins must be a divisor of n_steps.")
    size = int(n_steps / n_violins)
    # Determine x centers for the violins.
    x_centers = np.arange(start=0, stop=n_steps, step=size)
    # Group y values for each run.
    n_runs = n_runs if n_runs else max([data[key_run] for data in data_all]) + 1
    groups_list = []
    for run in range(n_runs):
        # Extract y values for the current run.
        yy_list = [data[key_y] for data in data_all if data[key_run] == run]
        groups_list += [[yy_list[ii:ii+size] for ii in range(0, n_steps, size)]]
    # Merge all data into a single run.
    if merge_data_all is True:
        groups_list = [np.concatenate(np.array(groups_list), axis=1).tolist()]
    # Plot data as half violins and dots.
    for groups in groups_list:
        # Plot half violins.
        parts = ax.violinplot(
            dataset=groups,
            positions=x_centers,
            widths=size * 1.5,
            side="high",
            **kwargs_violin,
        )
        # Customize violins.
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(edgecolor)
            pc.set_alpha(alpha_fill)
        # Plot top points as dots.
        for y_values, xx in zip(groups, x_centers):
            if scatter_top_n is not None:
                y_values = sorted(y_values, reverse=True)[:scatter_top_n]
            x_values = [xx] * len(y_values)
            ax.scatter(x=x_values, y=y_values, **kwargs_scatter)
    # Get x and y max for the plot.
    x_max = x_max if x_max else n_steps
    y_max = y_max if y_max else max([data[key_y] for data in data_all]) * 1.10
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    # Set the number of x ticks.
    if n_x_ticks is not None:
        ax.locator_params(axis="x", nbins=n_x_ticks)
        ax.grid(True, axis="x", linestyle="--", color="gray", alpha=0.5)
        ax.set_axisbelow(True)
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