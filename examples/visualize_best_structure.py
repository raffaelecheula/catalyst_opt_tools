# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
from ase.gui.gui import GUI
from ase.io.animation import write_animation

from catalyst_opt_tools.utilities import update_atoms_list

from reaction_rate_calculation import get_atoms_from_template_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = True
    write_gif = False

    # Parameters.
    miller_index = "100" # 100 | 111
    search_name = "GeneticAlgorithm" # Name of the search method.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"

    # Get data from yaml results file.
    data_all = yaml.safe_load(open(filename_yaml, "r"))

    # Get best structure from all runs.
    data_best = sorted(data_all, key=lambda xx: xx["rate"], reverse=True)[0]
    symbols_best = data_best["symbols"]

    # Get atoms from template database.
    atoms_list, n_atoms_surf = get_atoms_from_template_db(miller_index=miller_index)

    # Update elements of adsorbate atoms.
    update_atoms_list(
        atoms_list=atoms_list,
        symbols=symbols_best,
        n_atoms_surf=n_atoms_surf,
        update_features=False,
    )

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
            interval=50,
        )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------