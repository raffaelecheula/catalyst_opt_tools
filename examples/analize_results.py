# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ase.gui.gui import GUI
from ase.io.animation import write_animation

from ase_ml_models.yaml import write_to_yaml
from catalyst_opt_tools.utilities import update_atoms_list, print_title
from catalyst_opt_tools.plots import plot_cumulative_max_curve, plot_half_violins

from reaction_rate_calculation import get_atoms_from_template_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    plot_distr = True
    plot_cumul = False

    # Parameters.
    miller_index = "100" # 100 | 111
    search_name = "ScikitOptimization" # Name of the search method.
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