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
    n_eval = 500 # Number of structures evaluated per run.
    n_runs = 1 # Number of search runs.
    random_seed = 42 # Random seed for reproducibility.
    search_name = "CondVarAutoEnc" # Name of the search method.

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
        "n_random_samples": 200,
        "delta_y_cond": 0.,
        "latent_dim": 32,
        "hidden_dim_1": 128,
        "hidden_dim_2": 64,
        "optimizer_kwargs": {"lr": 1e-4},
        "n_epochs": 1000,
    }
    
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
    
    # Run multiple searches.
    data_all = []
    for run_id in range(n_runs):
        print_title(f"{search_name}: Run {run_id}")
        data_run = run_cvae_search(
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
# RUN CVAE SEARCH
# -------------------------------------------------------------------------------------

def run_cvae_search(
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
    Run a CVAE search.
    """
    import random
    # Get parameters for initial random search.
    n_random_samples = search_kwargs.pop("n_random_samples")
    delta_y_cond = search_kwargs.pop("delta_y_cond")
    # Prepare data storage for the run.
    data_run = []
    # Random search of surface with highest reaction rate.
    random.seed(random_seed+run_id)
    for jj in range(n_random_samples):
        # Get elements for the surface.
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        # Calculate reaction rate.
        rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        data_run.append({"symbols": symbols, "rate": rate, "run": run_id})
        # Print results to screen.
        if print_results is True:
            print(f"Symbols =", ",".join(symbols))
            print(f"Reaction Rate = {rate:+7.3e} [1/s]")
    # Extract maximum rate. We use all the data because we use conditioning.
    y_cond = max([data["rate"] for data in data_run]) + delta_y_cond
    # Get dataloader from data list.
    dataloader = get_dataloader_from_data_list(
        data_list=data_run,
        element_pool=element_pool,
    )
    # Initialize the CVAE model.
    n_elements = len(element_pool)
    model = CVAE(n_atoms_surf=n_atoms_surf, n_elements=n_elements, **search_kwargs)
    # Train the CVAE model.
    model.train_model(dataloader=dataloader)
    # Generate new samples using the trained CVAE model.
    generated_samples = model.generate_new_samples(
        n_samples=n_eval-n_random_samples,
        y_cond=y_cond,
    )
    # Evaluate generated samples and calculate reaction rates.
    for sample in generated_samples:
        # Get elements for the surface.
        symbols = get_symbols_from_tensor(tensor=sample, element_pool=element_pool)
        # Calculate reaction rate.
        rate = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        data_run.append({"symbols": symbols, "rate": rate, "run": run_id})
        # Print results to screen.
        if print_results is True:
            print(f"Symbols =", ",".join(symbols))
            print(f"Reaction Rate = {rate:+7.3e} [1/s]")
    # Get best structure.
    if print_results is True:
        data_best = sorted(data_run, key=lambda xx: xx["rate"], reverse=True)[0]
        rate_best, symbols_best = data_best["rate"], data_best["symbols"]
        print(f"Best Structure of run {run_id}:")
        print(f"Symbols =", ",".join(symbols_best))
        print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")
    # Return run data.
    return data_run

# -------------------------------------------------------------------------------------
# CVAE
# -------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CVAE(nn.Module):
    def __init__(
        self,
        n_atoms_surf: int,
        n_elements: int,
        latent_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        optimizer: object = optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-4},
        n_epochs: int = 100,
        loss_type: str = "CE", # CE | BCE | MSE
        kl_weight: float = 1e-2,
        final_activation: str = "softmax", # softmax | sigmoid | softmax_per_atom
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
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.n_epochs = n_epochs
        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.final_activation = final_activation
        self.cond_dim = 1
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
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        # Apply activation.
        if self.final_activation == "sigmoid":
            decoded = torch.sigmoid(decoded)
        elif self.final_activation == "softmax":
            decoded = torch.softmax(decoded, dim=-1)
        elif self.final_activation == "softmax_per_atom":
            decoded = self.softmax_per_atom(decoded)
        # Return decoded, mu and logvar.
        return decoded, mu, logvar

    def compute_loss(self, recon_x, x, mu, logvar):
        """
        Compute the CVAE loss function.
        """
        # Reconstruction Loss.
        if self.loss_type == "CE":
            loss = nn.functional.cross_entropy(recon_x, x, reduction="sum")
        elif self.loss_type == "BCE":
            loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        elif self.loss_type == "MSE":
            loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        # KL divergence Loss.
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Return loss.
        return loss + self.kl_weight * kl

    def train_model(self, dataloader):
        """ 
        Train the CVAE model.
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
        Generate new surface configurations using the trained CVAE model.
        """
        self.eval()
        # Sample random points from the latent space.
        zz = torch.randn(n_samples, self.latent_dim)
        # Generate y_cond if it is none or constant.
        if y_cond is None:
            y_cond = torch.zeros(n_samples, 1)
        elif isinstance(y_cond, float):
            y_cond = torch.full((n_samples, 1), y_cond)
        # Get z_cond.
        z_cond = torch.cat([zz, y_cond], dim=1)
        generated_samples = self.decoder(z_cond)
        # Return generated samples.
        return generated_samples

    def softmax_per_atom(self, x):
        """
        Applies softmax to each atom's element group independently.
        Assumes x has shape [batch_size, n_atoms * n_elements].
        """
        batch_size, total_dim = x.shape
        n_atoms = total_dim // self.n_elements
        x = x.view(batch_size, n_atoms, self.n_elements)
        x = torch.softmax(x, dim=-1)
        return x.view(batch_size, total_dim)

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
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------