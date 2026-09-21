# $\mathcal{W}$- and $\mathcal{M}$-information

A Python package for calculating $\mathcal{W}$- and $\mathcal{M}$-information.

<div align="center">
  <img src="wimfo_logo.png" alt="Alt text" width="50%">
</div>

## Description

$\mathcal{W}$- and $\mathcal{M}$-information are scalable measures of lower- and higher-order (beyond-pairwise) information in complex dynamical systems, respectively. They can be computed via a convex optimisation problem that scales gracefully with system size, enabling applications to large complex systems. $\mathcal{M}$-information is robust to noise, indexes critical dynamics in neural populations, and tracks consciousness levels and task performance in real brain activity. It can also be incorporated into existing information decomposition frameworks ($\Phi$ID) to reveal a finer taxonomy of information dynamics in multivariate time series.

## Installation

```bash
git clone https://github.com/alberto-liardi/wimfo.git
cd wimfo
./install.sh
```

This installs the bundled private dependencies (`pytorch-minimize`, `gpid`, `dit`) followed by the `wimfo` package itself, via `pip install -e .` (see `setup.py`).

## Getting started

See `examples.ipynb` for full worked examples. In short:

```python
from wimfo.W_M_Info import W_M_calculator

# Gaussian data (variables x samples matrix)
W, M = W_M_calculator(data, type="gaussian", option="data")

# discrete data, with a given alphabet size
W, M = W_M_calculator(data, type="discrete", option="data", alphabet_size=2)
```

`W_M_calculator` also accepts a covariance matrix or probability distribution directly (`option="distr"`), and can compute pointwise $\mathcal{W}$- and $\mathcal{M}$-information (`pointwise=True`). See the docstring in `wimfo/W_M_Info.py` for the full list of options (optimiser choice, units, future lag, etc.).

For the Partial Information Decomposition built on top of $\mathcal{W}$- and $\mathcal{M}$-information, use:

```python
from wimfo.Broja_PhiID import PhiID

atoms = PhiID(data, type="gaussian")
```

and similarly for discrete data (`type="discrete"`). 

## Repository structure

- `wimfo/` — the core library;
  - `W_M_Info.py` — contains the wrapper function `W_M_calculator`, which computes $\mathcal{W}$- and $\mathcal{M}$-information from data or a distribution (Gaussian or discrete);
  - `Broja_PhiID.py` — computes Integrated Information Decomposition ($\Phi$ID) based on $\mathcal{W}$- and $\mathcal{M}$-information;
  - `gaussian/`, `discrete/` — contains the estimators specialised for Gaussian and discrete systems;
  - `utils/` — includes shared helper functions
- `examples.ipynb` — contains worked examples showing how to use the library;
- `private/` - repository dependencies used by `wimfo`.

## Citation

If you use this package, please cite:

> Alberto Liardi, George Blackburne, Hardik Rajpal, Fernando E. Rosas, Pedro A.M. Mediano,
> *A scalable estimator of higher-order information in complex dynamical systems*,
> Cell Reports Physical Science, 2026, 103550, ISSN 2666-3864,
> https://doi.org/10.1016/j.xcrp.2026.103550
> ([sciencedirect.com/science/article/pii/S266638642600456X](https://www.sciencedirect.com/science/article/pii/S266638642600456X))

## License

BSD 3-Clause License, see `LICENSE`.
