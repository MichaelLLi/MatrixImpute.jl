# MatrixImpute
 
This is the Julia Repository for general low-rank Matrix Completion (with and without Side Information). The main interface is:

`Impute(A,k;method=:fastImpute,B=nothing, kwargs)`


- `A` is the Matrix of size `n x m` with missing entries. The entries missing should be of type Missing.
- `k` is the desired integer rank of fitting.
- `B` is the side information matrix of size `m x p`. If it is not specified, then it is assumed there is no side information.
- `kwargs` are named optional arguments to specify hyperparameters for each method. For the specific names of the hyperparameters, please see the methods listed below.
- `method` is the keyword for specifying the low rank imputation method. Currently the following methods are implemented:
   - fastImpute ([Fast Exact Matrix Completion: A Unifying Optimization Framework](https://arxiv.org/abs/1910.09092). The hyperparameters are: `lr` (learning rate) and `γ` (regularization parameter).
   - (Not Completed) softImpute ([Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares] (https://arxiv.org/pdf/1410.2596.pdf)) The hyperparameter arguments are: `λ` (regularization parameter). 

Current version is tested on Julia v1.2.

R package `fastImpute` would be available soon.
