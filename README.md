# NNLSSPAR
>A Python Package to solve sparsity constrained non-negative least squares problems

This repository contains a package to solve the following problem

[![g](https://github.com/Fatih-S-AKTAS/NNLSSPAR/blob/master/files/nnlssparquestion.png)]()

**Assumptions**
- System is overdetermined, A is a m x n matrix where m > n.
- A has full rank, rank(A) = n

# Usage

This package uses guppy3 by  YiFei Zhu and Sverker Nilsson for tracking memory usage. Hence guppy3 must be installed prior to using LSSPAR. 

```python
pip install guppy3
```

Then, after downloading LSSPAR.py, it can be used as follows:

1. Register values of matrix A, vector b and integer s.
2. Create instance of LSSPAR
3. Call solve function

```python

from NNLS_SPAR import * 

A = # feature (design) Matrix
b = # vector of variable to be predicted
s = # sparsity level

question = NNLSSPAR(A,b,s)

question.solve()
```

more details can be found in <a href="--guide link---" target="_blank">Guide for NNLSSPAR</a>
