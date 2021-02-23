A collection of scipy multivariate single criteria optimization solver
examples are located in this repository.

As for now, there presented following examples:
- [Broyden-Fletcher-Goldfarb-Shano with loss function gradient provided.](bfgs_grad.py)
- [Broyden-Fletcher-Goldfarb-Shano with no loss function gradient provided.](bfgs_nograd.py)
The gradient will be estimated with finite differences.
- [Nelder-Mead algorithm. No gradient needed.](nelder_mead.py)
This algorithm requires only the definition of function been optimized.
