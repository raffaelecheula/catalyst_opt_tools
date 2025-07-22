# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

from ase_ml_models.yaml import read_atoms_from_yaml
from ase_ml_models.databases import write_atoms_list_to_db
from catalyst_opt_tools.adsorption import (
    get_cluster_from_surface,
    get_surface_edges,
    get_adsorption_sites,
    get_bidentate_sites,
    get_sites_directions,
    adsorption_monodentate,
    adsorption_bidentate,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = True
    write_to_db = True

    # Surface type.
    miller_index = "100" # 100 | 111 | 211

    # Select species.
    species_list = ["CO*", "H*", "O*", "OH*", "H2O*", "CO2**"]
    
    # Get periodic surface.
    if miller_index == "100":
        from ase.build import fcc100
        atoms_periodic = fcc100(symbol="Au", size=(3, 3, 4), vacuum=10.0)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "111":
        from ase.build import fcc111
        atoms_periodic = fcc111(symbol="Au", size=(3, 3, 4), vacuum=10.0)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "211":
        from ase.build import fcc211
        atoms_periodic = fcc211(symbol="Au", size=(6, 3, 4), vacuum=10.0)
        indices_site = [0, 1, 7, 10, 15, 16]
    # Highlight site atoms.
    for ii in indices_site:
        atoms_periodic[ii].symbol = "Cu"
    
    # Get cluster from periodic surface.
    atoms_surf = get_cluster_from_surface(
        atoms=atoms_periodic,
        method="ase",
        bond_cutoff=1,
        indices_ads=[],
        indices_site=indices_site,
        remove_pbc=True,
        skin=0.20,
    )
    atoms_surf.info["species"] = "clean"
    atoms_surf.info["indices_ads"] = []
    indices_surf = atoms_surf.info["indices_site"]
    # Get edges from connectivity.
    edges_surf = get_surface_edges(
        connectivity=atoms_surf.info["connectivity"],
        indices_surf=indices_surf,
    )
    # Get mono-dentate adsorption sites.
    sites_dict = get_adsorption_sites(
        indices_surf=indices_surf,
        edges_surf=edges_surf,
    )
    # Get bi-dentate adsorption sites.
    sites_bi_dict = get_bidentate_sites(
        sites_dict=sites_dict,
    )
    sites_dict.update(sites_bi_dict)
    # Get directions of adsorption sites.
    directions = get_sites_directions(
        atoms_surf=atoms_surf,
        sites_dict=sites_dict,
    )
    
    # Read molecules.
    atoms_mol_list = read_atoms_from_yaml(filename="molecules.yaml")
    # Filter molecules by species.
    atoms_mol_list = [
        atoms for atoms in atoms_mol_list if atoms.info["species"] in species_list
    ]
    
    # Prepare adsorbates.
    atoms_surfads_list = []
    for atoms_mol in atoms_mol_list:
        surf_bound = atoms_mol.info["surf_bound"]
        sites_names = atoms_mol.info["sites_names"]
        for site_name in sites_names:
            # Get atoms for the site.
            for site_indices in sites_dict[site_name]:
                if len(surf_bound) == 1:
                    # Mono-dentate adsorption.
                    atoms_surfads = adsorption_monodentate(
                        atoms_mol=atoms_mol,
                        atoms_surf=atoms_surf,
                        surf_bound=surf_bound,
                        site_indices=site_indices,
                    )
                    atoms_surfads_list.append(atoms_surfads)
                elif len(surf_bound) == 2:
                    # Bi-dentate adsorption.
                    atoms_surfads = adsorption_bidentate(
                        atoms_mol=atoms_mol,
                        atoms_surf=atoms_surf,
                        surf_bound=surf_bound,
                        site_indices=site_indices,
                    )
                    atoms_surfads_list.append(atoms_surfads)
                # Update atoms info.
                atoms_surfads.info.update({
                    "species": atoms_mol.info["species"],
                    "site_name": site_name,
                })
    
    # Show atoms in GUI.
    if show_atoms is True:
        gui = GUI(atoms_surfads_list)
        gui.run()

    # Write atoms to database.
    if write_to_db is True:
        db_ase_name = f"databases/atoms_adsorbates_{miller_index}_templates.db"
        db_ase = connect(db_ase_name, append=False)
        atoms_list = [atoms_surf] + atoms_surfads_list
        write_atoms_list_to_db(atoms_list=atoms_list, db_ase=db_ase)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------