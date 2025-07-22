# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.workflow import (
    update_ts_atoms,
    get_atoms_ref,
    get_crossvalidator,
    crossvalidation,
    calibrate_uncertainty,
    parity_plot,
    violin_plot,
    groups_errors_plot,
    uncertainty_plot,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Cross-validation parameters.
    species_type = "adsorbates" # adsorbates | reactions # Type of species.
    miller_index = "111" # 100 | 111 # Miller index of the surface.
    stratified = True # Stratified cross-validation.
    group = False # Group cross-validation.
    key_groups = "surface" # surface | elements # Key for grouping the data.
    key_stratify = "species" # Key for stratification.
    n_splits = 5 # Number of splits for cross-validation.
    random_state = 42 # Random state for reproducibility.
    ensemble = False # Use the cross-validator to get an ensemble of models.
    resampling = False # Use resampling (bootstrapping) to get an ensemble of models.
    n_resamples = 100 # Number of samples for resampling.
    store_data = False # Store the data in an Ase database.
    add_ref_atoms = False # Add reference atoms to the training set.
    exclude_add = True # Exclude the reference atoms in the errors evaluation.
    filename_dist = None # f"distances_{species_type}.npy"
    
    # Model selection.
    model_name = "Graph" # Graph | PyG
    update_features = False # Update features of TS atoms from an Ase database.
    model_name_ref = model_name[:] # Reference model name for updating features.
    
    # Models parameters.
    species_ref = ["CO*", "H*", "O*"]
    # Get model parameters.
    if model_name == "Graph":
        # Graph model parameters.
        model_params = {
            "target": "E_form",
            "model_name": "KRR",
            "kwargs_kernel": {"length_scale": 30},
            "kwargs_model": {"alpha": 1e-4},
        }
    elif model_name == "PyG":
        # PyG model parameters.
        model_params = {
            "target": "E_form",
        }
    
    # Read Ase database.
    #db_ase_name = f"databases/atoms_{species_type}_DFT_database.db"
    db_ase_name = f"databases/atoms_{species_type}_{miller_index}_DFT.db"
    db_ase = connect(db_ase_name)
    kwargs = {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
    # Reference atoms to add to the train sets.
    if add_ref_atoms is True and species_type == "adsorbates":
        atoms_add = get_atoms_ref(atoms_list=atoms_list, species_ref=species_ref)
    else:
        atoms_add = []
    
    # Update TS features from an Ase database.
    if update_features is True and species_type == "reactions":
        db_ads_name = f"databases/atoms_adsorbates_{model_name_ref}_database.db"
        db_ads = connect(db_ads_name)
        update_ts_atoms(atoms_list=atoms_list, db_ads=db_ads)
    
    # Preprocess the data.
    if model_name == "Graph":
        from ase_ml_models.graph import graph_preprocess, precompute_distances
        node_weight_dict = {"A0": 1.00, "S1": 0.80, "S2": 0.20}
        edge_weight_dict = {"AA": 0.50, "AS": 1.00, "SS": 0.50}
        graph_preprocess(
            atoms_list=atoms_list,
            node_weight_dict=node_weight_dict,
            edge_weight_dict=edge_weight_dict,
        )
        distances = precompute_distances(atoms_X=atoms_list, filename=filename_dist)
        model_params.update({"distances": distances})
    
    # Print number of atoms.
    print(f"n atoms: {len(atoms_list)}")
    print(f"n added: {len(atoms_add)}")
    
    # Initialize cross-validation.
    crossval = get_crossvalidator(
        stratified=stratified,
        group=group,
        n_splits=n_splits,
        random_state=random_state,
    )
    # Prepare Ase database.
    db_model_name = f"databases/atoms_{species_type}_{model_name}_database.db"
    db_model = connect(db_model_name, append=False) if store_data else None
    # Cross-validation.
    results = crossvalidation(
        atoms_list=atoms_list,
        model_name=model_name,
        crossval=crossval,
        key_groups=key_groups,
        key_stratify=key_stratify,
        atoms_add=atoms_add,
        exclude_add=exclude_add,
        db_model=db_model,
        model_params=model_params,
        ensemble=ensemble,
        resampling=resampling,
        n_resamples=n_resamples,
    )
    
    # Plots parameters.
    plot_parity = True # Parity plot of predicted energies vs DFT energies.
    plot_species = False # Violin plots of errors distinguished by species.
    plot_materials = False # Violin plots of errors distinguished by material.
    plot_uncertainty = False # Parity plot of uncertainty vs error.
    # Colors and names for plots.
    color_dict = {"Graph": "crimson", "PyG": "orchid"}
    color = color_dict[model_name]
    task = f"groupval_{key_groups}" if group is True else "crossval"
    dirname = f"images/{task}_{miller_index}"
    os.makedirs(dirname, exist_ok=True)
    if species_type == "reactions" and update_features is True:
        model_name = f"{model_name}_from_{model_name_ref}"
    # Parity plot.
    if plot_parity is True:
        lims = [-2.2, +2.8] if species_type == "adsorbates" else [-1.4, +5.2]
        ax = parity_plot(results=results, lims=lims, color=color)
        plt.savefig(f"{dirname}/parity_{species_type}_{model_name}.png")
    # Species error plot.
    if plot_species is True:
        ax = groups_errors_plot(results, atoms_list, key="species", color=color)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
        plt.savefig(f"{dirname}/species_{species_type}_{model_name}.png")
    # Material error plot.
    if plot_materials is True:
        ax = groups_errors_plot(results, atoms_list, key="material", color=color)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
        plt.savefig(f"{dirname}/material_{species_type}_{model_name}.png")
    # Uncertainty quantification.
    if plot_uncertainty is True and "y_std" in results:
        results = calibrate_uncertainty(results=results, fit_intercept=False)
        ax = uncertainty_plot(results=results, color=color)
        plt.savefig(f"{dirname}/uncertainty_{species_type}_{model_name}.png")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    print(f"Execution time: {timeit.default_timer() - start:.2f} s")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------