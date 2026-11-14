# Theory: the model M t = p and matrix decompositions

`M` is a T × r matrix: a row is a timestamp, a column is a process, and an element is the memory
consumed by that process at that timestamp. The vector `p` of length T holds the total measured memory.
The vector `t` of length r holds unknown process weights. The model `M t = p` looks for `t` such that
the weighted sum of the process profiles matches the measurement.

If `T > r` and the columns of `M` are linearly independent, the least-squares solution is unique.
Then QR gives `t = R^{-1} Q^T p` with `M = Q R`, where `Q` has orthonormal columns and `R` is upper
triangular. If the columns are dependent, `R` is singular and QR cannot give a solution. The SVD
`M = U diag(s) V^T` then gives the minimum-norm solution `V diag(1/s) U^T p`, where singular values
below `rcond` are dropped.

The normal equations `(M^T M + λ I) t = M^T p` are solved by Cholesky factorisation. A positive ridge `λ`
makes the system solvable when `M` is rank deficient. The price is that the condition number is squared.

If the process weights must be non-negative, the problem `min ||M t - p||` with `t >= 0` (NNLS) is solved
by projected gradient descent with step `1/L`, where `L` is the largest eigenvalue of `M^T M`.

Non-negative matrix factorisation (NMF) finds `M ~ W H` with `W >= 0` and `H >= 0`. Columns of `W` can be
read as latent profiles and rows of `H` as their shares in each process. The factors come from Lee and Seung
multiplicative updates. The factorisation is not unique, so the result depends on the start `seed`.

The package does not guarantee that recovered weights match real processes. That is the hypothesis of the
experiment, not a property of the decompositions. See `НОВИЗНА.md` and `АНАЛИЗ.md` of the article folder.
