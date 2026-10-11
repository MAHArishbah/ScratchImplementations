# Logistic Regression

Binary classification from scratch, in two notebooks that differ in exactly one respect:
whether the classes can be separated by a straight line. The first fits a linear decision
boundary; the second reaches a curved one by expanding the features to degree-6 polynomials
and adding L2 regularisation to keep that expansion from overfitting.

## Files

| File | Description |
|---|---|
| [`LogisticRegression_Linear_DecisionBoundary.ipynb`](LogisticRegression_Linear_DecisionBoundary.ipynb) | Linearly separable case — plain logistic regression, boundary plotted in both scaled and original units |
| [`LogregNLDB.ipynb`](LogregNLDB.ipynb) | Non-linear case — polynomial feature map, regularised cost and gradient, contour-drawn boundary |
| `ex2data1.txt` | 100 rows, `exam1_score, exam2_score, admitted` — university admissions |
| `ex2data2.txt` | 118 rows, `test1, test2, accepted` — microchip QA, classes arranged in a ring |

Both notebooks share the same core: a `sigmoid`, the cross-entropy cost
`J(θ) = −Σ[y·log h + (1−y)·log(1−h)] / n`, and batch gradient descent on
`θ ← θ − α·Xᵀ(h − y)/n`. The gradient has the same algebraic form as linear regression's,
which is worth noticing — it falls out of the log-loss and sigmoid pairing, not by
coincidence.

In both, `h` is clipped to `[1e-7, 1−1e-7]` before the log. Without that, a confident
correct prediction saturates the sigmoid to exactly 0 or 1 and the cost becomes `inf`.

## Linear decision boundary

Features are z-score standardised, then a bias column is prepended (after scaling, so the
intercept is untouched). Trained at `alpha = 10` for 100 iterations — a step size that
large is only usable *because* of the standardisation.

The boundary is the line `θ₀ + θ₁x₁ + θ₂x₂ = 0`, solved for `x₂` and drawn through two
points, since a line needs no more. It is plotted twice:

- **in standardised space**, where the algebra is immediate, and
- **in original exam-score units**, by mapping the two endpoint `x₁` values forward through
  the scaler, solving for `x₂` there, then inverting back with `x₂·σ + μ`.

That second plot is the one worth the effort: a model trained on standardised features
still has to make statements about real-world quantities, and the round trip through the
scaler is where sign and offset errors show up.

A single prediction (`[45, 85]` → admission probability) is run through the same
scale-then-prepend-bias path, which is the correct inference pipeline for a new point.

## Non-linear decision boundary

The scatter plot makes the problem obvious — the accepted chips form a blob surrounded by
rejected ones, so no line can separate them. The response is to change the feature space
rather than the model.

**`poly_features(x1, x2, degree)`** builds every monomial `x₁^(i−j)·x₂^j` for `i` up to
`degree` and `j ≤ i`, producing all 28 terms up to degree 6 (including the leading bias
column). Logistic regression stays linear *in its parameters*; the boundary becomes curved
because the features are.

**Scaling with 28 correlated columns matters more here**, so the bias column is split off,
only the genuine features are standardised (with `ddof=1`), and the bias is reattached.

**`compute_cost_grad` adds L2 regularisation** — `(λ/2n)·Σθ²` on the cost and `(λ/n)·θ` on
the gradient, both **excluding `θ₀`**. The intercept is deliberately left unpenalised:
shrinking it biases the decision threshold rather than controlling model complexity. The
function returns cost and gradient together, since both need the same `h`. Trained at
`alpha = 0.5`, `λ = 1`, 3000 iterations.

**The boundary is drawn as a contour**, which is the general method: evaluate `θᵀφ(u, v)`
over a 200×200 grid — pushing each grid point through the *same* polynomial map and the
*same* scaler — and let `plt.contour(..., levels=[0])` trace where the decision function
crosses zero. No closed-form rearrangement needed, and it works for any boundary shape.

`lamb` is the knob to experiment with: raising it smooths the boundary toward a line,
lowering it toward 0 lets the degree-6 surface wrap individual points.

## Running them

```
jupyter notebook LogisticRegression_Linear_DecisionBoundary.ipynb
jupyter notebook LogregNLDB.ipynb
```

Paths are relative, so run from this directory. Requires `numpy` and `matplotlib`.
