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
    search_name = "TorchCVAE" # Name of the search method.

    # Results files.
    filename_yaml = f"results/{search_name}_{miller_index}.yaml"
    filename_png = f"results/{search_name}_{miller_index}_distr.png"

    # Input data.
    filename_input = f"results/DualAnnealing_{miller_index}.yaml"
    n_input = 500

    # Parameters for the search.
    search_kwargs = {
        "mult_y_cond": 0.,
        "n_random_samples": 0,
        "n_top_selected": 1000,
        "latent_dim": 32,
        "hidden_dim_1": 128,
        "hidden_dim_2": 64,
        "optimizer_kwargs": {"lr": 1e-3},
        "n_epochs": 100,
        "loss_type": "CEPA",
        "kl_weight": 0.1,
        "final_activation": "gumbel_softmax_per_atom",
        "gumbel_temperature": 1.0,
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
        data_run = run_generative_CVAE_model(
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
# RUN GENERATIVE CVAE MODEL
# -------------------------------------------------------------------------------------

def run_generative_CVAE_model(
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
    filename_yaml: str = "TorchCVAE.yaml",
    search_kwargs: dict = {},
    data_input: list = None,
):
    """
    Run a structure optimization with a generative PyTorch CVAE model.
    """
    import random
    random.seed(random_seed)
    # Pop parameters from search kwargs.
    search_kwargs = search_kwargs.copy()
    n_random_samples = search_kwargs.pop("n_random_samples", 0)
    n_top_selected = search_kwargs.pop("n_top_selected", 100)
    mult_y_cond = search_kwargs.pop("mult_y_cond", 1.)
    # Prepare data storage for the run.
    data_run = data_input or []
    if write_results is True and len(data_run) > 0:
        write_to_yaml(filename=filename_yaml, data=data_run, mode="a")
    # Run initial random search.
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
    # Initialize the CVAE model.
    model = CVAE(
        n_atoms_surf=n_atoms_surf,
        n_elements=len(element_pool),
        **search_kwargs,
    )
    # Extract the data with higher rate.
    data_list = sorted(data_run, key=lambda data: data["rate"])[-n_top_selected:]
    # Extract maximum rate.
    y_cond = max([data["rate"] for data in data_list]) * mult_y_cond
    # Get dataloader from data list.
    dataloader = get_dataloader_from_data_list(
        data_list=data_list,
        element_pool=element_pool,
    )
    # Train the CVAE model.
    model.train_model(dataloader=dataloader)
    # Generate new samples using the trained CVAE model.
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
# CVAE
# -------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class CVAE(nn.Module):
    def __init__(
        self,
        n_atoms_surf: int,
        n_elements: int,
        latent_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        cond_dim: int = 1,
        optimizer: object = Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        n_epochs: int = 100,
        loss_type: str = "CEPA",
        kl_weight: float = 0.1,
        final_activation: str = "gumbel_softmax_per_atom",
        gumbel_temperature: float = 1.0,
    ):
        """
        Conditional Variational AutoEncoder (CVAE) model.
        """
        super(CVAE, self).__init__()
        self.n_atoms_surf = n_atoms_surf
        self.n_elements = n_elements
        self.input_dim = n_atoms_surf * n_elements
        self.latent_dim = latent_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.cond_dim = cond_dim
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.n_epochs = n_epochs
        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.final_activation = final_activation
        self.gumbel_temperature = gumbel_temperature
        # Define the encoder network.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.cond_dim, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_2, self.latent_dim * 2),
        )
        # Define the decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_dim, self.hidden_dim_2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_2, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, self.input_dim),
        )
    
    def forward(self, x, y):
        """
        Forward pass through the CVAE.
        """
        # Concatenate x and y for encoder.
        y_expand = y.expand(-1, 1) if y.ndim == 1 else y
        x_cond = torch.cat([x, y_expand], dim=1)
        # Encode.
        encoded = self.encoder(x_cond)
        mu, logvar = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        # Reparameterize.
        z = self.reparameterize(mu, logvar)
        # Concatenate z and y for decoder.
        z_cond = torch.cat([z, y_expand], dim=1)
        # Decode.
        decoded = self.decoder(z_cond)
        # Apply final activation.
        decoded = self.apply_final_activation(decoded)
        # Return decoded, mu and logvar.
        return decoded, mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def apply_final_activation(self, decoded):
        """
        Apply final activation to the decoded output.
        """
        if self.final_activation == "sigmoid":
            decoded = torch.sigmoid(decoded)
        elif self.final_activation == "softmax":
            decoded = torch.softmax(decoded, dim=-1)
        elif self.final_activation == "softmax_per_atom":
            decoded = self.softmax_per_atom(decoded)
        elif self.final_activation == "gumbel_softmax_per_atom":
            decoded = self.gumbel_softmax_per_atom(decoded, self.gumbel_temperature)
        return decoded

    def softmax_per_atom(self, x):
        """
        Applies softmax per element group.
        """
        batch_size, total_dim = x.shape
        n_atoms = total_dim // self.n_elements
        x = x.view(batch_size, n_atoms, self.n_elements)
        x = torch.softmax(x, dim=-1)
        return x.view(batch_size, total_dim)

    def gumbel_softmax_per_atom(self, x, temperature):
        """
        Applies Gumbel–Softmax sampling per atom group.
        """
        batch_size, total_dim = x.shape
        n_atoms = total_dim // self.n_elements
        x = x.view(batch_size, n_atoms, self.n_elements)
        # Gumbel noise.
        g = -torch.log(-torch.log(torch.rand_like(x) + 1e-10) + 1e-10)
        y = nn.functional.softmax((x + g) / temperature, dim=-1)
        return y.view(batch_size, total_dim)

    def compute_loss(self, recon_x, x, mu, logvar):
        """
        Compute the loss function.
        """
        # Reconstruction Loss.
        if self.loss_type == "CE":
            # Cross-entropy loss.
            loss = nn.functional.cross_entropy(recon_x, x, reduction="sum")
        elif self.loss_type == "BCE":
            # Binary cross-entropy loss.
            loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        elif self.loss_type == "MSE":
            # Mean squared error loss.
            loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        elif self.loss_type == "CEPA":
            # Cross-entropy between one-hot target and predicted distribution.
            loss = -(x * torch.log(recon_x + 1e-10)).sum()
        # KL divergence Loss.
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Return loss.
        return loss + self.kl_weight * kl

    def train_model(self, dataloader):
        """
        Train the model.
        """
        # Prepare optimizer.
        opt = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        # Train model.
        n_data = len(dataloader.dataset)
        self.train()
        for epoch in range(self.n_epochs):
            train_loss = 0.
            for x, y in dataloader:
                # Preparation.
                opt.zero_grad()
                # Forward pass.
                recon_x, mu, logvar = self(x, y)
                # Compute loss.
                loss = self.compute_loss(recon_x, x, mu, logvar)
                loss.backward()
                opt.step()
                # Accumulate loss.
                train_loss += loss.item()
            # Print training loss for the epoch.
            print(
                f"Epoch {epoch+1:4d}/{self.n_epochs}, Loss: {train_loss/n_data:7.4f}"
            )
            
    def generate_new_samples(self, n_samples, y_cond=None):
        """
        Generate new surface configurations using the trained model.
        """
        self.eval()
        # Sample random points from the latent space.
        z = torch.randn(n_samples, self.latent_dim)
        # Generate y_cond if it is none or constant.
        if y_cond is None:
            y_cond = torch.zeros(n_samples, 1)
        elif isinstance(y_cond, float):
            y_cond = torch.full((n_samples, 1), y_cond)
        # Concatenate z and y_cond.
        z_cond = torch.cat([z, y_cond], dim=1)
        # Decode the points to get new surface configurations.
        decoded = self.decoder(z_cond)
        # Apply final activation.
        generated_samples = self.apply_final_activation(decoded)
        # Return generated samples.
        return generated_samples

# -------------------------------------------------------------------------------------
# GET DATALOADER FROM DATA LIST
# -------------------------------------------------------------------------------------

def get_dataloader_from_data_list(
    data_list: list,
    element_pool: list,
    key_y: str = "rate",
    key_X: str = "symbols",
    batch_size: int = 1,
    use_log_y: bool = False,
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
    # Apply log to y.
    if use_log_y is True:
        y_list = [np.log(np.clip(yy, a_min=1e-12, a_max=None)) for yy in y_list]
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