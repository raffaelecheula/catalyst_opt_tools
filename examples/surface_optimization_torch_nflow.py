# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

from ase_ml_models.yaml import write_to_yaml
from catalyst_opt_tools.optimization import print_search_results, print_search_progress
from catalyst_opt_tools.plots import plot_half_violins
from catalyst_opt_tools.utilities import print_title, get_data_input_from_yaml

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

    # Parameters.
    miller_index = "100" # 100 | 111
    element_pool = ["Rh", "Cu", "Au", "Ni", "Pd", "Co"] # Possible surface elements.
    n_eval = 1000 # Number of structures evaluated per run.
    n_runs = 1 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "TorchNFlow" # Name of the search method.

    # Results files.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"
    filename_png = f"results/{search_name}_{miller_index}_distr.png"

    # Input data.
    filename_input = f"results/RandomSearch_{miller_index}.yaml"
    n_input = 300

    # Parameters for the search.
    search_kwargs = {
        "n_random_samples": 0,
        "delta_y_cond": 0.,
        "n_epochs": 100,
    }

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
    
    # Parameters for reaction rate evaluation.
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
    data_run_list = []
    for run_id in range(n_runs):
        print_title(f"{search_name}: Run {run_id}")
        # Run search.
        data_run = run_generative_NFlow_model(
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
    
# -------------------------------------------------------------------------------------
# RUN GENERATIVE NFLOW MODEL
# -------------------------------------------------------------------------------------

def run_generative_NFlow_model(
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    run_id: int,
    random_seed: int,
    write_results: bool = True,
    print_results: bool = True,
    print_progress: bool = False,
    filename_yaml: str = "TorchNFlow.yaml",
    search_kwargs: dict = {},
    data_input: list = None,
):
    """
    Run a structure optimization with a generative PyTorch NFlow model.
    """
    import random
    random.seed(random_seed)
    # Pop parameters from search kwargs.
    search_kwargs = search_kwargs.copy()
    n_random_samples = search_kwargs.pop("n_random_samples", 0)
    delta_y_cond = search_kwargs.pop("delta_y_cond", 0.)
    # Prepare data storage for the run.
    data_run = data_input or []
    if write_results is True and len(data_run) > 0:
        write_to_yaml(filename=filename_yaml, data=data_run, mode="a")
    # Random search of surface with highest reaction rate.
    for jj in range(n_random_samples):
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
    # Extract maximum rate. We use all the data because we use conditioning.
    y_cond = max([data["rate"] for data in data_run]) + delta_y_cond
    # Get dataloader from data list.
    dataloader = get_dataloader_from_data_list(
        data_list=data_run,
        element_pool=element_pool,
    )
    # Initialize the NFlow model.
    n_elements = len(element_pool)
    model = ConditionalFlow(
        n_atoms_surf=n_atoms_surf,
        n_elements=n_elements,
        **search_kwargs,
    )
    # Train the NFlow model.
    model.train_model(dataloader=dataloader)
    # Generate new samples using the trained NFlow model.
    generated_samples = model.generate_new_samples(
        n_samples=n_eval-len(data_run),
        y_cond=y_cond,
    )
    # Evaluate generated samples and calculate reaction rates.
    for sample in generated_samples:
        # Get elements for the surface.
        symbols = get_symbols_from_tensor(tensor=sample, element_pool=element_pool)
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
# CONDITIONAL FLOW
# -------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import gumbel_softmax
from torch.utils.data import DataLoader, TensorDataset
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import (
    AffineCouplingTransform,
    CompositeTransform,
    RandomPermutation,
)

class ConditionalFlow(nn.Module):
    def __init__(
        self,
        n_atoms_surf: int,
        n_elements: int,
        hidden_dim: int = 128,
        cond_dim: int = 1,
        n_layers: int = 6,
        optimizer: object = Adam,
        optimizer_kwargs: dict = {"lr": 1e-4},
        n_epochs: int = 100,
        use_gumbel: bool = True,
        tau: float = 1.0,
        hard: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_atoms_surf = n_atoms_surf
        self.n_elements = n_elements
        self.input_dim = n_atoms_surf * n_elements
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.n_epochs = n_epochs
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.hard = hard
        self.device = device
        # Build normalizing flow.
        def transform_net_create_fn(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features + cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_features),
            )
        # Build coupling flow.
        transforms = []
        mask = (torch.arange(0, self.input_dim) % 2).bool()
        for _ in range(self.n_layers):
            transforms.append(RandomPermutation(features=self.input_dim))
            transforms.append(AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=transform_net_create_fn,
            ))
        transform = CompositeTransform(transforms)
        base_dist = StandardNormal([self.input_dim])
        self.flow = Flow(transform, base_dist)

    def forward(self, x, y):
        """
        Compute log-prob of x given condition y.
        """
        return self.flow.log_prob(inputs=x, context=y)

    def train_model(self, dataloader):
        """
        Train the model.
        """
        # Prepare optimizer.
        opt = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        # Train model.
        n_data = len(dataloader.dataset)
        self.to(self.device).train()
        # Training loop.
        for epoch in range(self.n_epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # Apply Gumbel-Softmax for differentiable relaxation.
                if self.use_gumbel:
                    # Reshape into [batch, n_atoms, n_elements]
                    x = x.view(x.size(0), self.n_atoms_surf, self.n_elements)
                    # Apply Gumbel-Softmax.
                    x = gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=-1)
                    # Flatten back to [batch, n_atoms*n_elements]
                    x = x.view(x.size(0), -1)
                # Negative log-likelihood.
                loss = -self(x, y).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
            print(
                f"Epoch {epoch+1:3d}/{self.n_epochs}, Loss: {total_loss/n_data:.4f}"
            )

    def generate_new_samples(self, n_samples, y_cond):
        """
        Generate new surface configurations using the trained model.
        """
        self.eval()
        # Ensure y_cond shape [n_samples, cond_dim]
        if y_cond.dim() == 1:
            y_cond = y_cond.unsqueeze(0).expand(n_samples, -1)
        y_cond = y_cond.to(self.device)
        # Sample from the flow.
        z = self.flow.sample(n_samples, context=y_cond)
        # Apply Gumbel-Softmax.
        if self.use_gumbel:
            # Reshape into [batch, n_atoms, n_elements]
            z = z.view(n_samples, self.n_atoms_surf, self.n_elements)
            # Apply Gumbel-Softmax.
            z = gumbel_softmax(z, tau=self.tau, hard=self.hard, dim=-1)
            # Flatten back to [batch, n_atoms*n_elements]
            z = z.view(n_samples, -1)
        # Return samples.
        return z

# -------------------------------------------------------------------------------------
# GET DATALOADER FROM DATA LIST
# -------------------------------------------------------------------------------------

def get_dataloader_from_data_list(
    data_list: list,
    element_pool: list,
    key_y: str = "rate",
    key_X: str = "symbols",
    batch_size: int = 8,
):
    """
    Get a PyTorch DataLoader from a dictionary of data.
    """
    # Transform elements to one-hot encoded vectors.
    n_elements = len(element_pool)
    element_to_encoded = {
        el: torch.nn.functional.one_hot(torch.tensor(ii), num_classes=n_elements)
        for ii, el in enumerate(element_pool)
    }
    # Prepare data for DataLoader.
    X_list = [[element_to_encoded[el] for el in data[key_X]] for data in data_list]
    y_list = [data[key_y] for data in data_list]
    # Convert lists to tensors.
    X_tensor = torch.stack([torch.cat(struct) for struct in X_list]).float()
    y_tensor = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    # Create a TensorDataset and DataLoader.
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # Return the DataLoader.
    return dataloader

# -------------------------------------------------------------------------------------
# GET SYMBOLS FROM TENSOR
# -------------------------------------------------------------------------------------

def get_symbols_from_tensor(
    tensor,
    element_pool: list,
) -> list:
    """
    Convert a flat one-hot-like tensor into a list of element symbols.
    """
    n_elements = len(element_pool)
    symbols = []
    for ii in range(0, len(tensor), n_elements):
        onehot = tensor[ii:ii + n_elements]
        index = torch.argmax(onehot).item()
        symbols.append(element_pool[index])
    return symbols

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