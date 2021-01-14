# NNLSSPAR
>A Python Package to solve sparsity constrained non-negative least squares problems

This repository contains a package to solve the following problem

[![g](https://github.com/Fatih-S-AKTAS/NNLSSPAR/blob/master/files/nnlssparquestion.png)]()

**Assumptions**
- System is overdetermined, A is a m x n matrix where m > n.
- A has full rank, rank(A) = n

# Usage

This package uses guppy3 by  YiFei Zhu and Sverker Nilsson for tracking memory usage. Hence guppy3 must be installed prior to using NNLSSPAR. 

```python
pip install guppy3
```

Then, after downloading NNLSSPAR.py, it can be used as follows:

1. Register values of matrix A, vector b and integer s.
2. Create instance of NNLSSPAR
3. Call solve function

```python

from NNLS_SPAR import * 

A = # feature (design) Matrix
b = # vector of variable to be predicted
s = # sparsity level

question = NNLSSPAR(A,b,s)

question.solve()
```

It also allows extra variables which are not constrained. We call this new problem as "Extended Least Squares", more details can be found in <a href="https://github.com/Fatih-S-AKTAS/NNLSSPAR/blob/master/Guide%20for%20NNLSSPAR.pdf">Guide for NNLSSPAR</a>

If you are using NNLSSPAR in your project, please give reference.
