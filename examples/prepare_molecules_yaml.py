# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.gui.gui import GUI

from arkimede.catkit.build import molecule
from ase_ml_models.utilities import get_connectivity
from ase_ml_models.yaml import write_atoms_to_yaml

# -------------------------------------------------------------------------------------
# GET ATOMS MOL
# -------------------------------------------------------------------------------------

def get_atoms_mol(
    species: str,
    vacuum: float = 12.0,
):
    """ 
    Get atoms from molecule name.
    """
    # Get molecules from name.
    if species == "O*":
        surf_bound = [0]
        atoms_mol = molecule("O", bond_index=surf_bound)[0]
        sites_names = ["brg", "3fh", "4fh"]
        symmetric_ads = False
    elif species == "C*":
        surf_bound = [0]
        atoms_mol = molecule("C", bond_index=surf_bound)[0]
        sites_names = ["brg", "3fh", "4fh"]
        symmetric_ads = False
    elif species == "H*":
        surf_bound = [0]
        atoms_mol = molecule("H", bond_index=surf_bound)[0]
        sites_names = ["top", "brg", "3fh", "4fh"]
        symmetric_ads = False
    elif species == "CO*":
        surf_bound = [0]
        atoms_mol = molecule("CO", bond_index=surf_bound)[0]
        sites_names = ["top", "brg", "3fh", "4fh"]
        symmetric_ads = False
    elif species == "OH*":
        surf_bound = [0]
        atoms_mol = molecule("OH", bond_index=surf_bound)[0]
        sites_names = ["top", "brg", "3fh", "4fh"]
        symmetric_ads = False
    elif species == "H2O*":
        surf_bound = [0]
        atoms_mol = molecule("H2O", bond_index=surf_bound)[0]
        sites_names = ["top"]
        symmetric_ads = False
    elif species == "CO2**":
        surf_bound = [0, 1]
        atoms_mol = molecule("CO2", bond_index=surf_bound)[0]
        sites_names = ["top-top", "brg-brg", "top-brg", "brg-top"]
        symmetric_ads = False
    elif species == "CO2*":
        surf_bound = [1]
        atoms_mol = molecule("CO2", bond_index=surf_bound)[0]
        sites_names = ["top"]
        symmetric_ads = False
    elif species == "COH*":
        surf_bound = [0]
        atoms_mol = molecule("COH", bond_index=surf_bound)[1]
        sites_names = ["top", "brg", "3fh", "4fh"]
        symmetric_ads = False
    elif species == "HCO**":
        surf_bound = [0, 1]
        atoms_mol = molecule("HCO", bond_index=surf_bound)[0]
        sites_names = ["top-top", "brg-brg", "top-brg", "brg-top"]
        symmetric_ads = False
    elif species == "c-COOH**":
        surf_bound = [0, 2]
        atoms_mol = molecule("COOH", bond_index=surf_bound)[1]
        sites_names = ["top-top", "brg-brg", "top-brg", "brg-top"]
        symmetric_ads = False
    elif species == "t-COOH**":
        surf_bound = [0, 2]
        atoms_mol = molecule("COOH", bond_index=surf_bound)[1]
        atoms_mol.set_angle(0, 1, 3, -120)
        sites_names = ["top-top", "brg-brg", "top-brg", "brg-top"]
        symmetric_ads = False
    elif species == "HCOO**":
        surf_bound = [1, 2]
        atoms_mol = molecule("HCOO", bond_index=surf_bound)[0]
        sites_names = ["top-top", "brg-brg", "top-brg", "brg-top"]
        symmetric_ads = True
    # Get connectivity.
    connectivity = get_connectivity(
        atoms=atoms_mol,
        method="ase",
        skin=0.20,
    )
    # Center the molecule in vacuum.
    atoms_mol.center(vacuum / 2.0)
    # Update atoms info.
    atoms_mol.info = {
        "species": species,
        "surf_bound": surf_bound,
        "connectivity": connectivity,
        "sites_names": sites_names,
        "symmetric_ads": symmetric_ads,
    }
    # Return atoms.
    return atoms_mol

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    write_atoms = True
    show_atoms = True

    # List of adsorbates to prepare.
    adsorbates_list = [
        "H*",
        "O*",
        "CO*",
        "OH*",
        "H2O*",
        "CO2**",
        "COH*",
        "HCO**",
        "c-COOH**",
        "t-COOH**",
        "HCOO**",
    ]

    # Get atoms for each adsorbate.
    atoms_list = [
        get_atoms_mol(species=species, vacuum=12.) for species in adsorbates_list
    ]

    # Write atoms to yaml file.
    if write_atoms is True:
        write_atoms_to_yaml(
            atoms_list=atoms_list,
            filename="molecules.yaml",
        )
    
    # Show atoms in GUI.
    if show_atoms is True:
        gui = GUI(atoms_list)
        gui.run()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
