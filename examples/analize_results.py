# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml

from catalyst_opt_tools.plots import plot_cumulative_max_curve, plot_half_violins

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    plot_distr = True
    plot_cumul = False

    # Parameters.
    miller_index = "100" # 100 | 111
    search_name = "RandomSearch" # Name of the search method.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"
    filename_distr = f"results/{search_name}_{miller_index}_distr.png"
    filename_cumul = f"results/{search_name}_{miller_index}_cumul.png"

    # Get data from yaml results file.
    data_all = yaml.safe_load(open(filename_yaml, "r"))

    # Plot half violins.
    if plot_distr is True:
        plot_half_violins(data_all=data_all, filename=filename_distr)
    
    # Plot cumulative max curve.
    if plot_cumul is True:
        plot_cumulative_max_curve(data_all=data_all, filename=filename_cumul)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------