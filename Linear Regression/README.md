# Linear Regression

Least squares regression solved two ways on the same data — iteratively by batch gradient
descent, and exactly by the normal equations — so the iterative answer can be checked
against the closed-form one.

## Files

| File | Description |
|---|---|
| [`Linear reg_1D.ipynb`](Linear%20reg_1D.ipynb) | Univariate linear regression: gradient descent, normal equations, loss surface and descent path |
| `ex1data1.txt` | 97 rows, `population, profit` — the single-feature dataset the notebook trains on |
| `ex1data2.txt` | 47 rows, `sq_ft, bedrooms, price` — multivariate data, kept for the multi-feature path |

## What the notebook does

**Split and standardise.** A 90/10 train/test split by random permutation (seeded), then
z-score standardisation using *training* statistics only — the test set is transformed with
the training mean and standard deviation, never its own. With one feature this is
cosmetic, but it is the habit that matters once features have unequal scales. A bias column
of ones is prepended afterwards so it is never scaled.

**Pick a learning rate from the curvature, not by trial and error.** For squared error the
Hessian is constant, `H = XᵀX / m`, so its eigenvalues are available before training. The
notebook prints two bounds:

- `2 / λ_max` — the stability limit; above it gradient descent diverges.
- `2 / (λ_max + λ_min)` — the rate that minimises the worst-case convergence factor, which
  is the optimal constant step size for a quadratic.

**Solve it exactly.** `normalEquations` computes `θ = (XᵀX)⁻¹Xᵀy` as the ground truth, and
`computeCost` evaluates `J(θ) = Σ(Xθ − y)² / 2m` at that point. This gives the lowest
achievable training loss, so gradient descent's final cost has something to be measured
against rather than just "looking converged".

**Descend.** `GradientDescent` runs the full-batch update `θ ← θ − α·Xᵀ(Xθ − y)/m`, logging
train loss, test loss, and the full `θ` history at every iteration. The learning-rate
search is written as a grid (`alpha` is a list) so additional rates plot as extra curves on
the same axes.

## Plots and what to read from them

- **Loss vs. iteration**, one curve per learning rate — the shape diagnoses the step size:
  smooth and still falling means too small, jagged means too large.
- **Train vs. test loss** on shared axes, with the final values printed next to the normal-
  equation `θ` for direct comparison.
- **Residuals vs. predictions** with a zero line. Structure here (curvature, a fan shape)
  would indicate a violated model assumption; for a correct linear fit the band should look
  flat and unpatterned.
- **Fitted line** over the scatter, with points sorted by `x` before plotting — necessary
  because the data was shuffled by the split, and `plt.plot` on unsorted points draws a
  zig-zag rather than a line.
- **Loss surface** over a `(θ₀, θ₁)` grid as a 3-D surface, plus a log-spaced contour plot
  with the gradient descent path drawn on top from its start point to the converged
  minimum. The contours are ellipses — their eccentricity is the conditioning that the
  eigenvalue bounds above quantified.

## Running it

```
jupyter notebook "Linear reg_1D.ipynb"
```

Paths are relative, so run it from this directory. Requires `numpy`, `matplotlib`, and —
only for the multivariate pair plot — `pandas` and `seaborn`.
