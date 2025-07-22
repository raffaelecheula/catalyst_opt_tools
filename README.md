# catalyst_opt_tools
Tools to optimize catalytic surfaces and maximize reaction rates.

catalyst_opt_tools is a collection of computational tools and scripts designed to facilitate the optimization of catalytic surfaces. By systematically exploring the composition of these surfaces, the aim is to identify configurations that lead to significantly enhanced reaction rates and overall catalytic activity. This repository serves as a practical resource for researchers and engineers working on catalyst design and discovery.

## Features

- **Compositional Optimization**: Algorithms and scripts to explore various surface compositions.
- **Reaction Rate Prediction**: Tools to estimate or model reaction rates based on surface properties.
- **Surface Property Analysis**: Utilities for analyzing key characteristics of catalytic surfaces.
- **Data Handling**: Scripts for managing and processing input/output data related to catalytic systems.
- **Modular Design**: Components are designed to be easily integrated into existing workflows or extended for new research.

## Installation

To install the package, clone the repository and install it:

```bash
git clone https://github.com/raffaelecheula/catalyst_opt_tools.git
cd catalyst_opt_tools
pip install -e .
```

Requirements:
- Python 3.5 or later
- Numpy
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [scikit-learn](https://scikit-learn.org/)

## Usage

The package provides modules and functions to optimize surface composition and maximize reaction rates. Example scripts demonstrating typical workflows are available in the `examples` directory of the repository.

## Contributing

Contributions to `catalyst_opt_tools` are welcome. If you have suggestions, bug reports, or would like to contribute code, please open an issue or submit a pull request on the [GitHub repository](https://github.com/raffaelecheula/catalyst_opt_tools).

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](https://github.com/raffaelecheula/catalyst_opt_tools/LICENSE) file for details.

## Acknowledgments

`catalyst_opt_tools` utilizes the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/), a set of tools and Python modules for setting up, manipulating, running, visualizing, and analyzing atomistic simulations. We acknowledge the developers of ASE for providing a robust framework that supports this package.

For more information on ASE and its capabilities, refer to the [ASE documentation](https://wiki.fysik.dtu.dk/ase/). 