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
    element_pool = ["Rh", "Cu", "Au"] # Possible elements of the surface.
    n_eval = 50 # Number of structures evaluated per run.
    n_runs = 5 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "GeneticAlgorithm" # Name of the search method.

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
    search_kwargs = {
        "sol_per_pop": 10,
        "keep_parents": 2,
        "num_parents_mating": 5,
        "mutation_percent_genes": 10,
        "parent_selection_type": "rank",
        "crossover_type": "single_point",
        "mutation_type": "random",
    }
    
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
    
    # Run multiple searches.
    data_all = []
    for run_id in range(n_runs):
        print_title(f"{search_name}: Run {run_id}")
        data_run = run_genetic_algorithm(
            reaction_rate_fun=reaction_rate_of_RDS_from_symbols,
            reaction_rate_kwargs=reaction_rate_kwargs,
            element_pool=element_pool,
            n_atoms_surf=n_atoms_surf,
            n_eval=n_eval,
            run_id=run_id,
            random_seed=random_seed,
            print_results=print_results,
            search_kwargs=search_kwargs,
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
# RUN GENETIC ALGORITHM
# -------------------------------------------------------------------------------------

def run_genetic_algorithm(
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    run_id: int,
    random_seed: int,
    print_results: bool = True,
    search_kwargs: dict = {},
):
    """ 
    Run a Bayesian optimization.
    """
    from pygad import GA
    # Prepare data storage for the run.
    data_run = []
    # Calculate number of generations.
    num_generations = int(np.ceil(
        (n_eval - search_kwargs["sol_per_pop"]) / 
        (search_kwargs["sol_per_pop"] - search_kwargs["keep_parents"])
    ))
    # Convert elements list to index and back.
    index_to_element = {ii: el for ii, el in enumerate(element_pool)}
    n_elements = len(element_pool)
    # Fitness function.
    def fitness_func(ga_instance, solution, solution_idx):
        # Convert indices to element symbols.
        symbols = [index_to_element[int(ii)] for ii in solution]
        # Calculate reaction rate of the rate-determining step.
        rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        if len(data_run) < n_eval:
            data_run.append({"symbols": symbols, "rate": rate, "run": run_id})
        # Print results to screen.
        if print_results is True:
            print(f"Symbols =", ",".join(symbols))
            print(f"Reaction Rate = {rate:+7.3e} [1/s]")
        # Return the rate.
        return rate
    # Set up the Genetic Algorithm.
    ga_instance = GA(
        num_generations=num_generations,
        fitness_func=fitness_func,
        num_genes=n_atoms_surf,
        gene_type=int,
        init_range_low=0,
        init_range_high=n_elements,
        gene_space=list(range(n_elements)),
        random_mutation_min_val=0,
        random_mutation_max_val=n_elements-1,
        random_seed=random_seed+run_id,
        **search_kwargs,
    )
    # Run the Genetic Algorithm.
    ga_instance.run()
    # Get best structure.
    if print_results is True:
        solution, rate_best, _ = ga_instance.best_solution()
        symbols_best = [index_to_element[int(ii)] for ii in solution]
        print(f"Best Structure of run {run_id}:")
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