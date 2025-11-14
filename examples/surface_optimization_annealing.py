# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

from catalyst_opt_tools.optimization import run_searches, run_dual_annealing
from catalyst_opt_tools.utilities import print_title, get_data_input_from_yaml
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
    print_results = False
    print_progress = True
    parallel_method = "multiprocessing" # serial | multiprocessing

    # Parameters.
    miller_index = "100" # 100 | 111
    element_pool = ["Rh", "Cu", "Au", "Ni", "Pd", "Co"] # Possible surface elements.
    n_eval = 1000 # Number of structures evaluated per run.
    n_runs = 3 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "DualAnnealing" # Name of the search method.

    # Results files.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"
    filename_png = f"results/{search_name}_{miller_index}_distr.png"

    # Input data.
    filename_input = None
    n_input = 0

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
    data_input_list = get_data_input_from_yaml(
        filename_input=filename_input,
        n_input=n_input,
        n_runs=n_runs,
    )
    
    # Run multiple searches.
    print_title(f"{search_name}: {n_runs} Runs")
    data_all = run_searches(
        search_name=search_name,
        n_runs=n_runs,
        optimization_fun=run_dual_annealing,
        reaction_rate_fun=reaction_rate_of_RDS_from_symbols,
        reaction_rate_kwargs=reaction_rate_kwargs,
        element_pool=element_pool,
        n_atoms_surf=n_atoms_surf,
        n_eval=n_eval,
        random_seed=random_seed,
        write_results=write_results,
        print_results=print_results,
        print_progress=print_progress,
        filename_yaml=filename_yaml,
        search_kwargs=search_kwargs,
        data_input_list=data_input_list,
        parallel_method=parallel_method,
    )
    
    # Plot half violins.
    plot_half_violins(data_all=data_all, filename=filename_png)
    
    # Get best structure from all runs.
    data_best = sorted(data_all, key=lambda xx: xx["rate"], reverse=True)[0]
    rate_best, symbols_best = data_best["rate"], data_best["symbols"]
    print_title(f"{search_name}: Best Structure")
    print(f"Symbols =", ",".join(symbols_best))
    print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")
    
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