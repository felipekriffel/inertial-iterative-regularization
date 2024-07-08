# inertial-iterative-regularization
Project for testing inertial regularizations in a elliptical PDE parametric inverse problem.

Here, we are interested in exploring different iterative inverse problem solvers, using as case-study the PDE

```math
- \Delta u + c\cdot u = f, \text{in } \Omega, \\

```
```math
u = g, \text{in } \partial \Omega,
```
where the inverse problem consists in recovering the function $c$ by knowing $u$ and $f$. This is a classic benchmark problem, which is used at this project for availing some iterative methods.
