# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.gui.gui import GUI

from ase_ml_models.yaml import write_to_yaml
from catalyst_opt_tools.utilities import update_atoms_list, print_title, parallel_runs
from catalyst_opt_tools.plots import plot_cumulative_max_curve

from reaction_rate_calculation import (
    get_graph_model_parameters,
    get_features_bulk_and_gas,
    get_trained_graph_model,
    get_atoms_from_template_db,
    reaction_rate_of_RDS_from_symbols,
)
from surface_optimization_random import run_random_search

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = False
    print_results = True
    write_results = True

    # Parameters.
    miller_index = "100" # 100 | 111
    element_pool = ["Rh", "Cu", "Au"] # Possible elements of the surface.
    n_eval = 50 # Number of structures evaluated per run.
    n_runs = 5 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "RandomSearch" # Name of the search method.

    # Results files.
    directory_yaml = f"results/{search_name}_{miller_index}"
    filename_png = f"results/{search_name}_{miller_index}.png"

    # Get model parameters and features.
    model_params, preproc_params = get_graph_model_parameters()
    features_bulk, features_gas = get_features_bulk_and_gas()
    # Get trained graph model.
    model = get_trained_graph_model(
        miller_index=miller_index,
        features_bulk=features_bulk,
        features_gas=features_gas,
        model_params=model_params,
        preproc_params=preproc_params,
    )

    # Get atoms from template database.
    atoms_list, n_atoms_surf = get_atoms_from_template_db(miller_index=miller_index)
    
    # Parameters for the search.
    search_kwargs = {}
    
    # Parameters for reaction rate function.
    reaction_rate_kwargs = {
        "atoms_list": atoms_list,
        "features_bulk": features_bulk,
        "features_gas": features_gas,
        "n_atoms_surf": n_atoms_surf,
        "model": model,
        "model_params": model_params,
        "preproc_params": preproc_params,
        "miller_index": miller_index,
    }
    
    # Create directory for YAML files.
    if write_results is True:
        os.makedirs(name=directory_yaml, exist_ok=True)
    
    # Data from previous run.
    data_input_list = [None] * n_runs
    
    # Run multiple searches.
    args_list = []
    for run_id in range(n_runs):
        # Name of the YAML file for this run.
        filename_yaml = f"{directory_yaml}/{run_id:02d}.yaml"
        # Reset YAML file.
        if write_results is True:
            open(file=filename_yaml, mode="w").close()
        # Run optimization.
        args_list.append([
            reaction_rate_of_RDS_from_symbols,
            reaction_rate_kwargs,
            element_pool,
            n_atoms_surf,
            n_eval,
            run_id,
            random_seed+run_id,
            print_results,
            write_results,
            filename_yaml,
            search_kwargs,
            data_input_list[run_id],
        ])
    
    # Execute parallel runs.
    data_run_list = parallel_runs(
        function=run_random_search,
        args_list=args_list,
        method="multiprocessing",
        n_jobs=len(args_list),
        with_tqdm=False,
    )
    # Combine data from all runs.
    data_all = [data for data_run in data_run_list for data in data_run]
    
    # Plot cumulative maximum rate curve.
    plot_cumulative_max_curve(data_all=data_all, filename=filename_png)
    
    # Get best structure from all runs.
    data_best = sorted(data_all, key=lambda xx: xx["rate"], reverse=True)[0]
    rate_best, symbols_best = data_best["rate"], data_best["symbols"]
    print_title(f"{search_name}: Best Structure")
    print(f"Symbols =", ",".join(symbols_best))
    print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")
    
    # Update elements of adsorbate atoms.
    update_atoms_list(
        atoms_list=atoms_list,
        features_bulk=features_bulk,
        features_gas=features_gas,
        symbols=symbols_best,
        n_atoms_surf=n_atoms_surf,
    )
    # Show atoms.
    if show_atoms is True:
        gui = GUI(atoms_list)
        gui.run()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    # Run main and measure execution time.
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print(f"Execution Time = {time_stop-time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------