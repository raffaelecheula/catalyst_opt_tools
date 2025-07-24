# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import random
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.yaml import write_to_yaml
from ase_ml_models.graph import graph_train, graph_predict, graph_preprocess
from catalyst_opt_tools.utilities import update_atoms_list, preprocess_features

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = True

    # Parameters.
    miller_index = "111" # 100 | 111
    elements = ["Rh", "Cu", "Au"] # Elements of the surface.
    random_seed = 42 # Random seed for reproducibility.

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
    
    # Get elements for the surface.
    random.seed(random_seed)
    symbols = random.choices(population=elements, k=n_atoms_surf)
    # Calculate reaction rate of the rate-determining step (RDS).
    rate = reaction_rate_of_RDS_from_symbols(
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
    print(f"Symbols =", ",".join(symbols))
    print(f"Reaction rate = {rate:+7.3e} [1/s]")

    # Show atoms.
    if show_atoms is True:
        gui = GUI(atoms_list)
        gui.run()

# -------------------------------------------------------------------------------------
# GET GRAPH MODEL PARAMETERS
# -------------------------------------------------------------------------------------

def get_graph_model_parameters() -> tuple:
    """
    Get parameters for the model and preprocessing.
    """
    # Graph model settings.
    model_params = {
        "target": "E_form",
        "model_name": "KRR",
        "kwargs_kernel": {"length_scale": 30},
        "kwargs_model": {"alpha": 1e-4},
    }
    preproc_params = {
        "node_weight_dict": {"A0": 1.00, "S1": 0.80, "S2": 0.20},
        "edge_weight_dict": {"AA": 0.50, "AS": 1.00, "SS": 0.50},
        "preproc": None,
    }
    # Return parameters.
    return model_params, preproc_params

# -------------------------------------------------------------------------------------
# GET GRAPH MODEL PARAMETERS
# -------------------------------------------------------------------------------------

def get_features_bulk_and_gas() -> tuple:
    """
    Get features for bulk and gas phase.
    """
    # Read features from yaml files.
    features_bulk = yaml.safe_load(open("features_bulk.yaml", "r"))
    features_gas = yaml.safe_load(open("features_gas.yaml", "r"))
    # Preprocess features.
    features_bulk = preprocess_features(features_dict=features_bulk)
    features_gas = preprocess_features(features_dict=features_gas)
    # Return parameters.
    return features_bulk, features_gas

# -------------------------------------------------------------------------------------
# GET TRAINED GRAPH MODEL
# -------------------------------------------------------------------------------------

def get_trained_graph_model(
    miller_index: str,
    features_bulk: dict,
    features_gas: dict,
    model_params: dict,
    preproc_params: dict,
):
    """
    Get trained graph model for adsorption energy predictions.
    """
    # Read atoms from DFT database.
    db_dft = connect(f"databases/atoms_adsorbates_{miller_index}_DFT.db")
    atoms_dft_list = get_atoms_list_from_db(db_ase=db_dft)
    # Update features of atoms from DFT database.
    update_atoms_list(
        atoms_list=atoms_dft_list,
        features_bulk=features_bulk,
        features_gas=features_gas,
    )
    # Get the trained graph model.
    model = train_graph_model(
        atoms_train=atoms_dft_list,
        model_params=model_params,
        preproc_params=preproc_params,
    )
    # Return model.
    return model

# -------------------------------------------------------------------------------------
# GET ATOMS FROM TEMPLATE DATABASE
# -------------------------------------------------------------------------------------

def get_atoms_from_template_db(
    miller_index: str,
):
    """
    Get atoms from template database.
    """
    # Read atoms objects from templates database.
    db_ase = connect(f"databases/atoms_adsorbates_{miller_index}_templates.db")
    atoms_list = get_atoms_list_from_db(db_ase=db_ase)
    # Get number of atoms in the surface.
    n_atoms_surf = len([
        atoms for atoms in atoms_list if atoms.info["species"] == "clean"
    ][0])
    # Return the list of atoms objects.
    return atoms_list, n_atoms_surf

# -------------------------------------------------------------------------------------
# REACTION RATE OF RDS FROM SYMBOLS
# -------------------------------------------------------------------------------------

def reaction_rate_of_RDS_from_symbols(
    symbols: list,
    atoms_list: list,
    features_bulk: dict,
    features_gas: dict,
    n_atoms_surf: int,
    model: object,
    model_params: dict,
    preproc_params: dict,
    miller_index: str,
):
    """
    Get reaction rate of the RDS from the surface symbols.
    """
    # Update elements and features of atoms for predictions.
    update_atoms_list(
        atoms_list=atoms_list,
        features_bulk=features_bulk,
        features_gas=features_gas,
        symbols=symbols,
        n_atoms_surf=n_atoms_surf,
    )
    # Predict formation energies with a graph model.
    predict_with_graph_model(
        atoms_list=atoms_list,
        model=model,
        model_params=model_params,
        preproc_params=preproc_params,
    )
    # Calculate reaction rate of the rate-determining step.
    rate = reaction_rate_of_RDS(
        atoms_list=atoms_list,
        miller_index=miller_index,
    )
    # Return the reaction rate.
    return rate

# -------------------------------------------------------------------------------------
# TRAIN GRAPH MODEL
# -------------------------------------------------------------------------------------

def train_graph_model(
    atoms_train: list,
    model_params: dict,
    preproc_params: dict,
):
    """
    Get a trained graph model for adsorption energy predictions.
    """
    # Preprocess the data.
    graph_preprocess(
        atoms_list=atoms_train,
        **preproc_params,
    )
    # Train the Graph model.
    model = graph_train(
        atoms_train=atoms_train,
        **model_params,
    )
    # Return the model.
    return model

# -------------------------------------------------------------------------------------
# PREDICT WITH GRAPH MODEL
# -------------------------------------------------------------------------------------

def predict_with_graph_model(
    atoms_list: list,
    model: object,
    model_params: dict,
    preproc_params: dict,
):
    """
    Get formation energies with a graph model.
    """
    # Preprocess the data.
    graph_preprocess(
        atoms_list=atoms_list,
        **preproc_params,
    )
    # Predict test data.
    y_pred = graph_predict(
        atoms_test=atoms_list,
        model=model,
        **model_params,
    )
    # Update formation energies.
    for atoms, e_form in zip(atoms_list, y_pred):
        atoms.info["E_form"] = e_form

# -------------------------------------------------------------------------------------
# REACTION RATE OF RDS
# -------------------------------------------------------------------------------------

def reaction_rate_of_RDS(
    atoms_list: list,
    miller_index: str,
):
    """
    Calculate reaction rate of the rate-determining step (RDS).
    """
    from mikimoto import units
    from mikimoto.microkinetics import Species
    from mikimoto.thermodynamics import ThermoNASA7
    
    # Set temperature and pressure of the simulation.
    temperature_celsius = 500. # [C]
    temperature = units.Celsius_to_Kelvin(temperature_celsius) # [K]
    pressure = 1 * units.atm # [Pa]
    # Set molar fractions of gas species.
    gas_molfracs_inlet = {
        "CO2": 0.28,
        "H2": 0.28,
        "H2O": 0.02,
        "CO": 0.02,
        "N2": 0.40,
    }
    
    # Get the lowest formation energies of the adsorbate species.
    e_form_dict = {"(X)": 0.0}
    for atoms in atoms_list:
        species = atoms.info["species"].replace("**", "(X)").replace("*", "(X)")
        e_form = atoms.info["E_form"]
        if species not in e_form_dict or e_form < e_form_dict[species]:
            e_form_dict[species] = e_form

    # Get the formation energy of the RDS from BEP relations.
    delta_h = e_form_dict["CO(X)"] + e_form_dict["O(X)"] - e_form_dict["CO2(X)"]
    if miller_index == "100":
        e_act_RDS = 0.789 + 0.624 * delta_h # [eV]
    elif miller_index == "111":
        e_act_RDS = 1.220 + 0.655 * delta_h # [eV]
    e_form_dict["CO2(X) + (X) <=> CO(X) + O(X)"] = e_act_RDS + e_form_dict["CO2(X)"]
    
    # Get energy corrections dictionary.
    e_corr_dict = e_form_dict.copy()
    e_corr_dict.update({spec: 0.0 for spec in gas_molfracs_inlet})
    # Read the mechanism from the YAML file.
    mechanism_dict = yaml.safe_load(open("mechanism.yaml", "r"))
    # Get standard Gibbs free energies.
    g0_form_dict = {}
    for species_type in ["gas", "adsorbates", "reactions"]:
        for species_data in mechanism_dict[f"species-{species_type}"]:
            spec = Species(
                name=species_data["name"],
                thermo=ThermoNASA7(
                    temperature=temperature,
                    coeffs_NASA=species_data["thermo"]["data"][0],
                ),
            )
            if spec.name in e_corr_dict:
                spec.thermo.modify_energy(
                    e_corr_dict[spec.name] * (units.eV/units.molecule) # [J/kmol]
                )
                g0_form_dict[spec.name] = spec.thermo.Gibbs_std # [J/kmol]

    # Calculate reduced pressures.
    p_red = {
        species: gas_molfracs_inlet[species] * pressure / units.atm
        for species in gas_molfracs_inlet
    } # [-]
    # Calculate the Gibbs free energies of adsorption.
    g_ads_dict = {
        "CO(X)": g0_form_dict["CO(X)"] - g0_form_dict["CO"],
        "H(X)": g0_form_dict["H(X)"] - g0_form_dict["H2"] * 0.5,
        "O(X)": g0_form_dict["O(X)"] - (g0_form_dict["H2O"] - g0_form_dict["H2"]),
    } # [J/kmol]
    # Calculate equilibrium constants.
    k_eq_dict = {
        species: np.exp(-g_ads_dict[species] / (units.Rgas * temperature))
        for species in g_ads_dict
    } # [-]
    
    # Calculate coverage of free sites.
    coverage_free = 1 / (
        1 + 
        k_eq_dict["CO(X)"] * p_red["CO"] + 
        k_eq_dict["H(X)"] * p_red["H2"] ** 0.5 +
        k_eq_dict["O(X)"] * p_red["H2O"] / p_red["H2"]
    )
    # Calculate activation energy and kinetic constant of RDS.
    g0_act_RDS = (
        g0_form_dict["CO2(X) + (X) <=> CO(X) + O(X)"] - g0_form_dict["CO2"]
    ) # [eV]
    a_for = units.kB * temperature / units.hP # [1/s]
    k_for_RDS = a_for * np.exp(-g0_act_RDS / (units.Rgas * temperature)) # [1/s]
    # Calculate reaction rate.
    rate = k_for_RDS * p_red["CO2"] * coverage_free ** 2 # [1/s]
    # Return the reaction rate.
    return float(rate)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------