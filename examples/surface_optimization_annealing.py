# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.gui.gui import GUI

from ase_ml_models.yaml import write_to_yaml
from catalyst_opt_tools.utilities import update_atoms_list, print_title
from catalyst_opt_tools.plots import plot_cumulative_max_curve

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
    show_atoms = False
    print_results = True
    write_results = True

    # Parameters.
    miller_index = "100" # 100 | 111
    elements = ["Rh", "Cu", "Au"] # Elements of the surface.
    n_eval = 50 # Number of rate evaluations per run.
    n_runs = 5 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "DualAnnealing" # Name of the search method.

    # Results files.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"
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
    search_params = {}
    
    # Run multiple searches.
    data_all = []
    for run in range(n_runs):
        print_title(f"{search_name}: Run {run}")
        data_run = run_dual_annealing(
            reaction_rate_fun=reaction_rate_of_RDS_from_symbols,
            elements=elements,
            n_atoms_surf=n_atoms_surf,
            n_eval=n_eval,
            run=run,
            random_seed=random_seed,
            atoms_list=atoms_list,
            features_bulk=features_bulk,
            features_gas=features_gas,
            model=model,
            model_params=model_params,
            preproc_params=preproc_params,
            miller_index=miller_index,
            print_results=print_results,
            search_params=search_params,
        )
        # Append run data to all data.
        data_all += data_run
    
    # Write results to YAML file.
    if write_results is True:
        write_to_yaml(filename=filename_yaml, data=data_all)
    
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
# RUN DUAL ANNEALING
# -------------------------------------------------------------------------------------

def run_dual_annealing(
    reaction_rate_fun: callable,
    elements: list,
    n_atoms_surf: int,
    n_eval: int,
    run: int,
    random_seed: int,
    atoms_list: list,
    features_bulk: dict,
    features_gas: dict,
    model: object,
    model_params: dict,
    preproc_params: dict,
    miller_index: str,
    print_results: bool = True,
    search_params: dict = {},
):
    """ 
    Run a dual annealing.
    """
    from scipy.optimize import dual_annealing
    # Prepare data storage for the run.
    data_run = []
    # Define objective function.
    def objective_fun(xx):
        # xx is an array of floats, map to nearest integer.
        x_int = [int(round(ii)) for ii in xx]
        symbols = [elements[ii] for ii in x_int]
        # Calculate reaction rate of the rate-determining step.
        rate = reaction_rate_fun(
            symbols=symbols,
            atoms_list=atoms_list,
            features_bulk=features_bulk,
            features_gas=features_gas,
            n_atoms_surf=n_atoms_surf,
            model=model,
            model_params=model_params,
            preproc_params=preproc_params,
            miller_index=miller_index,
        )
        if len(data_run) < n_eval:
            data_run.append({"symbols": symbols, "rate": rate, "run": run})
        # Print results to screen.
        if print_results is True:
            print(f"Symbols =", ",".join(symbols))
            print(f"Reaction Rate = {rate:+7.3e} [1/s]")
        # Return negative rate.
        return -rate
    # Perform dual annealing optimization.
    bounds = [(0, len(elements)-1)] * n_atoms_surf
    result = dual_annealing(
        func=objective_fun,
        bounds=bounds,
        maxfun=n_eval,
        seed=random_seed+run,
    )
    # Get best structure.
    if print_results is True:
        indices = [int(round(ii)) for ii in result.x]
        symbols_best = [elements[ii] for ii in indices]
        rate_best = -result.fun
        print(f"Best Structure of run {run}:")
        print(f"Symbols =", ",".join(symbols_best))
        print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")
    # Return run data.
    return data_run

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------