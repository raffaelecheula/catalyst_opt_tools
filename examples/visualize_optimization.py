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
from catalyst_opt_tools.plots import plot_cumulative_max_curve

from reaction_rate_calculation import get_atoms_from_template_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = False
    write_gif = True

    # Parameters.
    run_id = 0 # ID of the run to visualize.
    miller_index = "100" # 100 | 111
    search_name = "DualAnnealing" # Name of the search method.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"

    # Get data from yaml results file.
    data_all = yaml.safe_load(open(filename_yaml, "r"))
    data_run = [data for data in data_all if data["run"] == run_id]

    # Get atoms from template database.
    atoms_surf = get_atoms_from_template_db(miller_index=miller_index)[0][0]

    # Get list of atoms generated.
    atoms_list = []
    for data in data_run:
        atoms_copy = atoms_surf.copy()
        atoms_copy.symbols = data["symbols"]
        atoms_list.append(atoms_copy)

    # Show atoms.
    if show_atoms is True:
        gui = GUI(atoms_list)
        gui.run()

    # Write gif.
    if write_gif is True:
        write_animation(
            filename="animation.gif",
            images=atoms_list,
            scale=100,
            maxwidth=200, 
            radii=0.9,
            interval=20,
        )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------