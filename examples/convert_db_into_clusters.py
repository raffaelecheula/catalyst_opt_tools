# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

from ase_ml_models.databases import (
    get_atoms_list_from_db,
    write_atoms_list_to_db,
)
from ase_ml_models.utilities import (
    get_connectivity,
    plot_connectivity,
    print_features_table,
)
from catalyst_opt_tools.adsorption import get_cluster_from_surface
from catalyst_opt_tools.utilities import get_features

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = True
    show_wrong = False
    write_to_db = True

    # Surface and species type.
    miller_index = "111" # 100 | 111
    species_type = "adsorbates" # adsorbates | reactions

    # Read atoms from database.
    db_old_name = f"databases/atoms_{species_type}_DFT_database.db"
    db_old = connect(db_old_name)
    atoms_old_list = get_atoms_list_from_db(db_ase=db_old)

    # Select species.
    if species_type == "adsorbates":
        species_list = ["CO2*", "CO*", "H*", "O*", "OH*", "H2O*"]
        indices_site = [27, 28, 30, 31]
    elif species_type == "reactions":
        species_list = ["CO2*→CO*+O*", "H2O*→OH*+H*", "OH*→O*+H*"]
        indices_site = [12, 13, 14, 15]
    
    # Reduce atoms list to the selected species and miller index.
    atoms_old_list = [
        atoms for atoms in atoms_old_list 
        if atoms.info["miller_index"] == miller_index
        and atoms.info["species"] in species_list
    ]
    
    # Read features from yaml files.
    features_bulk = yaml.safe_load(open("features_bulk.yaml", "r"))
    features_gas = yaml.safe_load(open("features_gas.yaml", "r"))
    
    # Custom cutoffs for right connectivity.
    cutoffs_dict = {"Au": 1.70, "Ni": 1.40, "Ga": 1.40}
    
    # Number of atoms in the metal clusters.
    n_atoms_dict = {"100": 21, "111": 22}
    
    # Reduce atoms.
    atoms_list_new = []
    for atoms_old in atoms_old_list:
        # Get reduced graph atoms.
        indices_ads = atoms_old.info["indices_ads"]
        atoms_new = get_cluster_from_surface(
            atoms=atoms_old,
            method="ase",
            bond_cutoff=1,
            indices_ads=indices_ads,
            indices_site=indices_site,
            cutoffs_dict=cutoffs_dict,
            remove_pbc=True,
            mult=1.00,
            skin=0.20,
        )
        # Update keys.
        for key in ["name_original", "species_original", "indices_original"]:
            del atoms_new.info[key]
        get_features(
            atoms=atoms_new,
            features_bulk=features_bulk,
            features_gas=features_gas,
        )
        if len(atoms_new)-len(indices_ads) == n_atoms_dict[miller_index]:
            atoms_list_new.append(atoms_new)
        elif show_wrong is True:
            atoms_new.edit()
    
    # Write atoms to database.
    if write_to_db is True:
        db_new_name = f"databases/atoms_{species_type}_{miller_index}_DFT.db"
        db_new = connect(db_new_name, append=False)
        write_atoms_list_to_db(
            atoms_list=atoms_list_new,
            db_ase=db_new,
        )
    
    # Show atoms.
    if show_atoms is True:
        gui = GUI(atoms_list_new)
        gui.run()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------