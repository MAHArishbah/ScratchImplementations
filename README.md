# Scratch Implementations

Classical machine learning algorithms implemented from first principles with NumPy (and
PyTorch autograd where gradients are the point, not the lesson), each paired with a
reference implementation — scikit-learn, `torch.optim`, or a closed-form solution — so the
scratch version can be checked rather than assumed correct.

Every notebook follows the same shape: derive the update rule, implement it, verify it
against a trusted baseline, then plot the behaviour that the maths predicts.

## Contents

| Topic | What it covers |
|---|---|
| [Linear Regression](Linear%20Regression/) | Univariate least squares: gradient descent vs. the normal equations, loss-surface and descent-path visualisation |
| [Logistic Regression](Logistic%20Regression/) | Binary classification with linear and polynomial decision boundaries, L2 regularisation |
| [Optimisers](Optimisers/) | SGD, momentum, RMSprop, Adam, AdamW written from scratch and verified bit-for-bit against `torch.optim` |
| [`pca.ipynb`](pca.ipynb) | Principal component analysis three ways — eigendecomposition, SVD, and kernel PCA — validated against `sklearn.decomposition.PCA` |

## `pca.ipynb`

The only notebook that lives at the root, because it has no accompanying dataset: it runs
on scikit-learn's bundled `iris` and `digits` data.

Three implementations of the same idea, each a class with the scikit-learn
`fit` / `transform` / `fit_transform` / `inverse_transform` interface:

**`PCAEig`** — the textbook route. Mean-centre `X`, form the covariance matrix
`S = XᶜᵀXᶜ / (n−1)`, and take its eigendecomposition with `np.linalg.eigh` (symmetric, so
`eigh` rather than `eig`). Eigenvectors are sorted by descending eigenvalue; the top `k`
become `components_` and the eigenvalues become `explained_variance_`. Direct, but it
squares the condition number of `X` by building `S` explicitly.

**`PCASVD`** — the numerically stable route, and what scikit-learn actually does. The
singular value decomposition of the centred data, `Xᶜ = USVᵀ`, gives the components as the
rows of `Vᵀ` and the variances as `s² / (n−1)`, with no covariance matrix formed. It also
applies the standard **sign convention**: singular vectors are only defined up to sign, so
each column is flipped to make its largest-magnitude entry positive. Without that, two
correct implementations disagree by arbitrary sign flips.

**`KernelPCA`** — PCA in an implicit feature space via an RBF kernel. Builds the Gram
matrix `K`, double-centres it with `H = I − 11ᵀ/n` (centring must happen in feature space,
where the mean is not available directly), eigendecomposes the result, and normalises the
eigenvectors by `1/√μ` so the feature-space components have unit norm. New points are
projected by centring their kernel rows consistently with the training centring.

### How correctness is established

- **Projections** — `PCAEig`, `PCASVD` and `sklearn.PCA` are compared on `iris` under
  `np.allclose(np.abs(...))`; absolute values, because of the sign ambiguity above.
- **Components** — on `digits` with 40 components, `|components_ @ ref.components_.T|` is
  asserted to have a unit diagonal, i.e. each scratch component is parallel to the
  corresponding scikit-learn one. This is stricter than comparing projections, which can
  agree while individual axes are rotated within a degenerate eigenvalue block.
- **Explained variance** — ratios are asserted equal to scikit-learn's at default
  tolerance.
- **Reconstruction error** — `reconstruction_error` reports per-element MSE, per-sample
  squared distance, Frobenius norm, and error relative to total variance, so a lossy
  round-trip can be quantified rather than eyeballed.

Scree plots for both datasets show per-component and cumulative explained variance for the
scratch and reference implementations overlaid — the curves coincide, which is the visual
form of the same claim.

## Datasets

The regression notebooks use the Coursera *Machine Learning* exercise data (`ex1*`, `ex2*`),
kept alongside the notebook that reads them. The optimiser and PCA notebooks generate or
load their data programmatically, so they run with no local files.

## Requirements

```
numpy  matplotlib  scikit-learn  pandas  seaborn  torch
```

`pandas` and `seaborn` are only needed for the multivariate plotting branch in the linear
regression notebook; `torch` only for the optimisers notebook.
