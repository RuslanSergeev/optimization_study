# Optimization examples
---

A collection of scipy multivariate single criteria optimization solver
examples are located in this repository.

---

## Optimization
As for now, there presented following examples:
- [SGD from scratch example](sgd_scratch.py)
A simple $\alpha \cdot cos(\beta\cdot x+\gamma)$ optimization problem considered in the example, but
it is capable to optimize any given loss function.
- [Broyden-Fletcher-Goldfarb-Shano with loss function gradient provided.](bfgs_grad.py)  
Need a fit of A*cos(x+phi)+b into a set of sensor measured data.  
- [Broyden-Fletcher-Goldfarb-Shano with no loss function gradient provided.](bfgs_nograd.py)  
Same problem considered:  
Need a fit of A*cos(x+phi)+b into a set of sensor measured data.  
The gradient will be estimated with finite differences.  
- [Nelder-Mead algorithm. No gradient needed.](nelder_mead.py)  
This algorithm requires only the definition of function been optimized.  
- [Trusted region constrained optimization](trust_constr.py)  
The example calculates a control signal for a set of actuators, to  
obtain desired control vector.  
- [SLSQP constrained optimization](slsqp.py)  
The same example as in trusted-region: need to find a control  
vector for a set of actuators.  
- [Lagrange coefficients solution](Lagrange_coeffs.py)  
The same problem. Much more efficient and shorter analitical solution.  
Also we define an variable penalization matrix, instead of Identity matrix  
of the previous examples.  

## Dimensionality reduction
- [Moons classification with sklearn kernelized PCA](PCA_moons_simple.py)  
A simple toy data classification using sklearn kernelized PCA.  
- [Kernelized PCA from scratch](PCA_moons_hard.py)  
The same dataset as in above. RBF PCA written from scratch.  

## Miscelaneous examples:
- [Numerical Jacobian computation with finite differences](Jacobian_2point.py)  
A simple straightforward algorithm for Jacobian estimation.
