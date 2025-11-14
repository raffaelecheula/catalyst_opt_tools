# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.gui.gui import GUI

from catalyst_opt_tools.optimization import run_random_search
from catalyst_opt_tools.utilities import update_atoms_list, print_title
from catalyst_opt_tools.plots import plot_half_violins

from reaction_rate_calculation import (
    get_graph_model_parameters,
    get_features_bulk_and_gas,
    get_trained_graph_model,
    get_atoms_from_template_db,
    reaction_rate_of_RDS_from_symbols,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    write_results = True
    print_results = True
    print_progress = False
    show_atoms = False

    # Parameters.
    miller_index = "100" # 100 | 111
    element_pool = ["Rh", "Cu", "Au", "Ni", "Pd", "Co"] # Possible surface elements.
    n_eval = 100 # Number of structures evaluated per run.
    n_runs = 3 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "RandomSearch" # Name of the search method.

    # Results files.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"
    filename_png = f"results/{search_name}_{miller_index}_distr.png"

    # Parameters for the search.
    search_kwargs = {}

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
    
    # Reset YAML file.
    if write_results is True:
        open(file=filename_yaml, mode="w").close()
    
    # Data from previous run.
    data_input_list = [None] * n_runs
    
    # Run multiple searches.
    data_run_list = []
    for run_id in range(n_runs):
        print_title(f"{search_name}: Run {run_id}")
        # Run search.
        data_run = run_random_search(
            reaction_rate_fun=reaction_rate_of_RDS_from_symbols,
            reaction_rate_kwargs=reaction_rate_kwargs,
            element_pool=element_pool,
            n_atoms_surf=n_atoms_surf,
            n_eval=n_eval,
            run_id=run_id,
            random_seed=random_seed + run_id,
            write_results=write_results,
            print_results=print_results,
            print_progress=print_progress,
            filename_yaml=filename_yaml,
            search_kwargs=search_kwargs,
            data_input=data_input_list[run_id],
        )
        # Append run data to list.
        data_run_list.append(data_run)
    # Combine data from all runs.
    data_all = [data for data_run in data_run_list for data in data_run]
        
    # Plot half violins.
    plot_half_violins(data_all=data_all, filename=filename_png)
    
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