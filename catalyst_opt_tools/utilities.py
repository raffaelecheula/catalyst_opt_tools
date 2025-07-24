# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------------------------
# GET FEATURES
# -------------------------------------------------------------------------------------

def get_features(
    atoms: Atoms,
    features_bulk: dict,
    features_gas: dict,
) -> None:
    """
    Get features from atoms.
    """
    species = atoms.info["species"]
    features_bulk_names = features_bulk["names"]
    features_gas_names = features_gas["names"]
    n_features_bulk = len(features_bulk_names)
    n_features_gas = len(features_gas_names)
    features = np.zeros((len(atoms), n_features_bulk + n_features_gas))
    for ii, atom in enumerate(atoms):
        if ii in atoms.info["indices_ads"]:
            features[ii, n_features_bulk:] = features_gas[species]
        else:
            features[ii, :n_features_bulk] = features_bulk[atom.symbol]
    atoms.info["features"] = features
    atoms.info["features_names"] = features_bulk_names + features_gas_names

# -------------------------------------------------------------------------------------
# UPDATE ATOMS LIST
# -------------------------------------------------------------------------------------

def update_atoms_list(
    atoms_list: Atoms,
    features_bulk: dict,
    features_gas: dict,
    symbols: list = None,
    n_atoms_surf: int = None,
) -> None:
    """
    Update atoms list with new symbols and calculate corresponding features.
    """
    # Update atoms list.
    for atoms in atoms_list:
        # Update surface elements.
        if symbols is not None:
            atoms.symbols[:n_atoms_surf] = symbols
        # Update features.
        get_features(
            atoms=atoms,
            features_bulk=features_bulk,
            features_gas=features_gas,
        )

# -------------------------------------------------------------------------------------
# PREPROCESS FEATURES
# -------------------------------------------------------------------------------------

def preprocess_features(
    features_dict: dict,
    preproc: object = MinMaxScaler(feature_range=(-1, +1)),
) -> dict:
    """
    Preprocess features with a Scikit-learn preprocessing model.
    """
    # Remove features names.
    features_names = features_dict.pop("names")
    # Fit the preprocessing model.
    preproc.fit(np.vstack([features for features in features_dict.values()]))
    # Transform the features.
    for name, features in features_dict.items():
        features_dict[name] = [float(ii) for ii in preproc.transform([features])[0]]
    # Restore features names.
    features_dict["names"] = features_names
    # Return preprocessed features.
    return features_dict

# -------------------------------------------------------------------------------------
# PRINT TITLE
# -------------------------------------------------------------------------------------

def print_title(
    string: str,
    width: int = 100,
) -> None:
    """
    Print title.
    """
    for text in ["-" * width, string.center(width), "-" * width]:
        print("#", text, "#")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------