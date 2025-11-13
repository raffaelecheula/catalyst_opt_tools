# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import sys
import numpy as np

from ase_ml_models.yaml import write_to_yaml
from catalyst_opt_tools.utilities import print_title, parallel_runs

# -------------------------------------------------------------------------------------
# RUN SEARCHES
# -------------------------------------------------------------------------------------

def run_searches(
    search_name: str,
    n_runs: int,
    optimization_fun: callable,
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    random_seed: int,
    write_results: bool,
    print_results: bool,
    print_progress: bool,
    filename_yaml: str,
    search_kwargs: dict,
    data_input_list: list,
    write_results_runs: bool = False,
    directory_yaml: str = None,
    parallel_method: str = "multiprocessing",
    n_jobs: int = None,
) -> list:
    """
    Run searches in parallel.
    """
    # Prepare directory to store the yaml files.
    directory_yaml = directory_yaml or os.path.splitext(filename_yaml)[0]
    if write_results_runs is True:
        os.makedirs(directory_yaml, exist_ok=True)
    # Run multiple searches.
    args_list = []
    filename_list = []
    for run_id in range(n_runs):
        # Name of the YAML file for this run.
        filename_list.append(f"{directory_yaml}/{run_id:02d}.yaml")
        # Reset YAML file.
        if write_results_runs is True:
            open(file=filename_list[run_id], mode="w").close()
        # Run optimization.
        args_list.append([
            reaction_rate_fun,
            reaction_rate_kwargs,
            element_pool,
            n_atoms_surf,
            n_eval,
            run_id,
            random_seed + run_id,
            write_results_runs,
            print_results,
            print_progress,
            filename_list[run_id],
            search_kwargs,
            data_input_list[run_id],
        ])
    # Prepare line to print the progress.
    if print_progress is True:
        print("")
    # Execute parallel runs.
    data_run_list = parallel_runs(
        function=optimization_fun,
        args_list=args_list,
        parallel_method=parallel_method,
        n_jobs=n_jobs or len(args_list),
    )
    # Combine data from all runs.
    data_all = [data for data_run in data_run_list for data in data_run]
    # Write results to yaml file.
    if write_results is True:
        write_to_yaml(filename=filename_yaml, data=data_all, mode="w")
    # Return all data.
    return data_all

# -------------------------------------------------------------------------------------
# PRINT SEARCH RESULTS
# -------------------------------------------------------------------------------------

def print_search_results(
    symbols: list,
    rate: float,
) -> None:
    """
    Print to screen the results of the search.
    """
    print(f"Symbols =", ",".join(symbols))
    print(f"Reaction Rate = {rate:+7.3e} [1/s]")

# -------------------------------------------------------------------------------------
# PRINT SEARCH PROGRESS
# -------------------------------------------------------------------------------------

def print_search_progress(
    run_id: int,
    nn: int,
    n_eval: int,
) -> None:
    """
    Print to screen the progress of the search.
    """
    if sys.stdout.isatty():
        sys.stdout.write("\033[1A")
        sys.stdout.write(f"\033[{run_id * 18 + 1}G")
    print(f"Run {run_id:3d} = {nn / n_eval * 100:3.0f} % |")

# -------------------------------------------------------------------------------------
# RUN RANDOM SEARCH
# -------------------------------------------------------------------------------------

def run_random_search(
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    run_id: int,
    random_seed: int,
    write_results: bool = True,
    print_results: bool = False,
    print_progress: bool = True,
    filename_yaml: str = "RandomSearch.yaml",
    search_kwargs: dict = {},
    data_input: list = [],
) -> list:
    """
    Run a structure optimization with the random search method.
    """
    import random
    random.seed(random_seed)
    # Prepare data storage for the run.
    data_run = data_input or []
    if write_results is True and len(data_run) > 0:
        write_to_yaml(filename=filename_yaml, data=data_run, mode="a")
    # Random search of surface with highest reaction rate.
    for jj in range(n_eval):
        # Get elements for the surface.
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        # Calculate reaction rate.
        rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        data = {"symbols": symbols, "rate": rate, "run": run_id}
        data_run.append(data)
        # Write results to yaml.
        if write_results is True:
            write_to_yaml(filename=filename_yaml, data=[data], mode="a")
        # Print results of the search.
        if print_results is True:
            print_search_results(symbols=symbols, rate=rate)
        # Print progress of the search.
        elif print_progress is True:
            print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
    # Get best structure.
    if print_results is True:
        data = sorted(data_run, key=lambda xx: xx["rate"], reverse=True)[0]
        rate, symbols = data["rate"], data["symbols"]
        print(f"Best Structure of Run {run_id}:")
        print_search_results(symbols=symbols, rate=rate)
    # Return run data.
    return data_run

# -------------------------------------------------------------------------------------
# RUN SCIKIT OPTIMIZATION
# -------------------------------------------------------------------------------------

def run_scikit_optimization(
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    run_id: int,
    random_seed: int,
    write_results: bool = True,
    print_results: bool = False,
    print_progress: bool = True,
    filename_yaml: str = "ScikitOptimization.yaml",
    search_kwargs: dict = {},
    data_input: list = None,
) -> list:
    """
    Run a structure optimization with scikit-optimize (skopt).
    """
    from skopt.optimizer import base_minimize
    from skopt.space import Categorical
    from skopt.utils import use_named_args
    # Prepare data storage for the run.
    data_run = data_input or []
    if write_results is True and len(data_run) > 0:
        write_to_yaml(filename=filename_yaml, data=data_run, mode="a")
    # Define the search space.
    space = [Categorical(element_pool, name=f"el_{ii}") for ii in range(n_atoms_surf)]
    # Objective function.
    @use_named_args(space)
    def objective_func(**kwargs):
        # Extract symbol list from kwargs.
        symbols = [kwargs[f"el_{ii}"] for ii in range(n_atoms_surf)]
        # Calculate reaction rate of the rate-determining step.
        rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        data = {"symbols": symbols, "rate": rate, "run": run_id}
        data_run.append(data)
        # Write results to yaml.
        if write_results is True:
            write_to_yaml(filename=filename_yaml, data=[data], mode="a")
        # Print results of the search.
        if print_results is True:
            print_search_results(symbols=symbols, rate=rate)
        # Print progress of the search.
        elif print_progress is True:
            print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
        # Return the negative rate.
        return -rate
    # Run the scikit optimization.
    result = base_minimize(
        func=objective_func,
        dimensions=space,
        n_calls=n_eval - 1,
        random_state=random_seed,
        **search_kwargs,
    )
    # Get best structure.
    symbols = [xx for xx in result.x]
    rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
    data = {"symbols": symbols, "rate": rate, "run": run_id}
    data_run.append(data)
    # Write results to yaml.
    if write_results is True:
        write_to_yaml(filename=filename_yaml, data=[data], mode="a")
    # Print results of the search.
    if print_results is True:
        print(f"Best Structure of Run {run_id}:")
        print_search_results(symbols=symbols, rate=rate)
    # Print progress of the search.
    elif print_progress is True:
        print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
    # Return run data.
    return data_run

# -------------------------------------------------------------------------------------
# RUN DUAL ANNEALING
# -------------------------------------------------------------------------------------

def run_dual_annealing(
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    run_id: int,
    random_seed: int,
    write_results: bool = True,
    print_results: bool = False,
    print_progress: bool = True,
    filename_yaml: str = "DualAnnealing.yaml",
    search_kwargs: dict = {},
    data_input: list = None,
) -> list:
    """
    Run a structure optimization with the dual annealing method.
    """
    from scipy.optimize import dual_annealing
    # Prepare data storage for the run.
    data_run = data_input or []
    if write_results is True and len(data_run) > 0:
        write_to_yaml(filename=filename_yaml, data=data_run, mode="a")
    # Define objective function.
    def objective_fun(xx):
        # xx is an array of floats, map to nearest integer.
        x_int = [int(round(ii)) for ii in xx]
        symbols = [element_pool[ii] for ii in x_int]
        # Calculate reaction rate of the rate-determining step.
        rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        if len(data_run) >= n_eval - 1:
            return -rate
        data = {"symbols": symbols, "rate": rate, "run": run_id}
        data_run.append(data)
        # Write results to yaml.
        if write_results is True:
            write_to_yaml(filename=filename_yaml, data=[data], mode="a")
        # Print results of the search.
        if print_results is True:
            print_search_results(symbols=symbols, rate=rate)
        # Print progress of the search.
        elif print_progress is True:
            print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
        # Return the negative rate.
        return -rate
    # Perform dual annealing optimization.
    bounds = [(0, len(element_pool)-1)] * n_atoms_surf
    result = dual_annealing(
        func=objective_fun,
        bounds=bounds,
        maxfun=n_eval - 1,
        seed=random_seed,
    )
    # Get best structure.
    indices = [int(round(xx)) for xx in result.x]
    symbols = [element_pool[ii] for ii in indices]
    rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
    data = {"symbols": symbols, "rate": rate, "run": run_id}
    data_run.append(data)
    # Write results to yaml.
    if write_results is True:
        write_to_yaml(filename=filename_yaml, data=[data], mode="a")
    # Print results of the search.
    if print_results is True:
        print(f"Best Structure of Run {run_id}:")
        print_search_results(symbols=symbols, rate=rate)
    # Print progress of the search.
    elif print_progress is True:
        print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
    # Return run data.
    return data_run

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
    write_results: bool = True,
    print_results: bool = False,
    print_progress: bool = True,
    filename_yaml: str = "GeneticAlgorithm.yaml",
    search_kwargs: dict = {},
    data_input: list = None,
) -> list:
    """
    Run a structure optimization with the genetic algorithm method.
    """
    from pygad import GA
    # Prepare data storage for the run.
    data_run = data_input or []
    if write_results is True and len(data_run) > 0:
        write_to_yaml(filename=filename_yaml, data=data_run, mode="a")
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
        if len(data_run) >= n_eval - 1:
            return rate
        data = {"symbols": symbols, "rate": rate, "run": run_id}
        data_run.append(data)
        # Write results to yaml.
        if write_results is True:
            write_to_yaml(filename=filename_yaml, data=[data], mode="a")
        # Print results of the search.
        if print_results is True:
            print_search_results(symbols=symbols, rate=rate)
        # Print progress of the search.
        elif print_progress is True:
            print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
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
        random_seed=random_seed,
        **search_kwargs,
    )
    # Run the Genetic Algorithm.
    ga_instance.run()
    # Get best structure.
    solution, rate_best, _ = ga_instance.best_solution()
    symbols = [index_to_element[int(ii)] for ii in solution]
    rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
    data = {"symbols": symbols, "rate": rate, "run": run_id}
    data_run.append(data)
    # Write results to yaml.
    if write_results is True:
        write_to_yaml(filename=filename_yaml, data=[data], mode="a")
    # Print results of the search.
    if print_results is True:
        print(f"Best Structure of Run {run_id}:")
        print_search_results(symbols=symbols, rate=rate)
    # Print progress of the search.
    elif print_progress is True:
        print_search_progress(run_id=run_id, nn=len(data_run), n_eval=n_eval)
    # Return run data.
    return data_run

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------