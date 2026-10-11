# Optimisers

Six gradient-based optimisers written from scratch on top of PyTorch autograd — SGD, SGD
with momentum, RMSprop, RMSprop with momentum, Adam, and AdamW — then verified against
`torch.optim` to float64 round-off and compared against each other on two deliberately
different problems.

## Files

| File | Description |
|---|---|
| [`optimisers.ipynb`](optimisers.ipynb) | All six implementations plus the three experiments below, with written interpretation of every result |
| `mlp_loss_curves.png`, `trajectories_2d.png` | Latest figures, kept at this level |
| `Optimiser_plots_<timestamp>/` | Per-run output directories, created automatically by `OUTPUT_DIR` on each execution |

The timestamped folders are generated artifacts, not source — each run of the notebook
creates a fresh one, so they are expected to accumulate and are safe to delete.

## The implementations

A small `Optimizer` base class holds the shared machinery: it materialises the parameter
generator into a list, keeps **one state dict per parameter** (so momentum and variance
buffers stay per-tensor), tracks a global step counter `t` for Adam's bias correction, and
exposes `zero_grad` / `step`. `step` is decorated with `@torch.no_grad()` and skips
parameters whose `.grad` is `None`; subclasses implement only `_update(p, g, state)`.

| Optimiser | Update rule |
|---|---|
| `SGD` | `θ ← θ − lr·g` |
| `SGDMomentum` | `v ← μv + g`;  `θ ← θ − lr·v` |
| `RMSProp` | `s ← αs + (1−α)g²`;  `θ ← θ − lr·g/(√s + ε)` |
| `RMSPropMomentum` | `s ← αs + (1−α)g²`;  `b ← μb + g/(√s + ε)`;  `θ ← θ − lr·b` |
| `Adam` | `m ← β₁m + (1−β₁)g`;  `v ← β₂v + (1−β₂)g²`;  bias-correct both;  `θ ← θ − lr·m̂/(√v̂ + ε)` |
| `AdamW` | Adam, but `θ ← θ(1 − lr·λ)` applied *outside* the adaptive rescaling |

Two details are load-bearing because they are where reimplementations usually drift from
the library:

**Momentum uses the PyTorch convention `v ← μv + g`**, with no `(1−μ)` factor. A
consistent gradient therefore accumulates to `g/(1−μ)`, so at `μ = 0.9` the effective step
is roughly **10× the nominal learning rate**. Matching this convention is exactly why the
implementations agree with `torch.optim`.

**Adam and AdamW differ only in where weight decay enters.** `Adam` adds `λθ` to the
gradient *before* the moments, so the penalty is then divided by `√v̂` — coordinates with a
large gradient history end up barely regularised, and regularisation strength becomes
coupled to gradient scale. `AdamW` multiplies `θ` by `(1 − lr·λ)` directly, shrinking every
coordinate by the same factor. For plain SGD the two schemes are algebraically identical
(`θ − lr(g + λθ) = (1 − lr·λ)θ − lr·g`); the distinction only becomes real once the update
is adaptive.

## Part 1 — correctness against `torch.optim`

`compare_with_torch` trains two models from **identical initial weights** on **identical
batch indices** — one with a scratch optimiser, one with its `torch.optim` counterpart —
and tracks `max |θ_mine − θ_torch|` over every parameter scalar at every one of 100 steps.
Not the loss: two different update rules can land on similar losses by coincidence, but
they cannot agree scalar-by-scalar for 100 consecutive steps.

`torch.set_default_dtype(torch.float64)` is what makes the test meaningful. In float32
(ε ≈ 1.2e-7) the `1e-10` tolerance is unreachable and the test would pass or fail for
reasons unrelated to the code.

Seven of eight cases agree to ~1e-16, one unit in the last place of float64. The tolerance
sits in the empty band between float64 noise (1e-16 … 1e-12) and the smallest plausible
algorithmic disagreement (≥ 1e-4). RMSprop + momentum lands at ~1e-12 — looser because it
is the deepest composition in the set and round-off entering `s` is amplified by a
recursive velocity buffer; a genuine formula error would show up at order `lr·|g| ≈ 1e-3`.

**The sanity check is the important line.** Eight passes are also what a broken,
insensitive test prints, so the notebook deliberately compares scratch
`Adam(weight_decay=0.1)` against `torch.optim.AdamW(weight_decay=0.1)` — different
algorithms — and confirms a difference of ~1.4. That is what proves the harness can detect
a real discrepancy.

Not covered: one seed, one architecture, one dtype, 100 steps, no LR schedule, no parameter
groups, no sparse gradients, no `state_dict` round-trip.

## Part 2 — the optimisers against each other (`mlp_loss_curve.png`)

A 2→32→32→1 tanh MLP (1185 parameters) on `make_moons(600, noise=0.2)`, standardised, for
60 epochs at batch 32. All optimisers share one seeded generator, so every one sees the
same batch ordering — a controlled comparison, which is why loss spikes line up across
methods at the same epochs.

Two configurations are run side by side:

- **`TUNED`** — a sensible learning rate per optimiser. All six finish within 0.093–0.108
  loss and 0.962–0.972 training accuracy. **Once the learning rate is tuned, the choice of
  optimiser barely matters on this problem**, which matches the careful benchmark
  literature (Choi et al. 2019; Schmidt et al. 2021). The ~0.09 floor is label noise from
  `noise=0.2`, not an optimisation failure — every method is sitting on the irreducible
  floor, so this column measures who *reaches* it, not who goes lower.
- **`SAME`** — `lr = 0.01` for all six. This is the informative column: it does not rank
  optimisers, it measures how each rule converts a nominal learning rate into a physical
  step size. SGD under-trains (0.295, smooth and still descending — the signature of too
  small a step, not divergence); SGD momentum recovers to 0.110 via the 10× effective-step
  multiplier; RMSprop and Adam are nearly unchanged from their tuned runs, because
  `g/√s ≈ ±1` makes the step ≈ `lr` by construction; RMSprop + momentum destabilises
  (‖θ‖ ≈ 105), having stacked momentum on an *already normalised* gradient.

The run also reports `‖θ‖`, the L2 norm of all weights. Adaptive methods finish with
systematically larger weights because they keep taking ≈ `lr`-sized steps even in flat,
saturated-tanh regions where the raw gradient has effectively vanished — SGD stops moving
when `|g| → 0`, adaptive methods do not. That is the practical argument for decoupled
weight decay.

Two caveats the notebook states explicitly: `TUNED["Adam"]` has `weight_decay = 0` while
`TUNED["AdamW"]` has `0.1`, so their ‖θ‖ gap shows *decay vs. no decay*, not *coupled vs.
decoupled*; and `acc` is training accuracy, supporting no claim about generalisation.

## Part 3 — trajectories on an ill-conditioned quadratic (`trajectories_2d.png`)

`f(x, y) = 0.1x² + 2y²`, started at `(−5, −2)` for 50 steps, each optimiser's path drawn
over the contours. The Hessian is the constant `diag(0.2, 4)`, so the condition number is
κ = 20 and the level sets are ellipses with axis ratio √κ ≈ 4.47 — a long narrow valley.

This isolates the problem adaptive methods exist to solve. Gradient descent on a quadratic
decouples per eigendirection with multiplier `1 − lr·λ`: stability along `y` demands
`lr < 2/λ_max = 0.5`, but any `lr` below that leaves the `x` multiplier at ≥ 0.9, so the
flat direction crawls. One scalar learning rate must serve two curvatures 20× apart, and
each subplot is a different answer to that.

SGD at `lr = 0.4` is predictable in closed form — the `y` multiplier is `1 − 1.6 = −0.6`,
negative, hence the visible per-step sawtooth that dies after ~15 steps, while
`x₅₀ = −5·0.92⁵⁰ = −0.077` matches the printed value exactly. RMSprop reaches ~1e-21,
best by many orders: diagonal preconditioning neutralises the curvature gap, and because
`s` shrinks along with `g`, the *relative* contraction per step stays constant instead of
slowing down.

The notebook's interpretation section is careful about two results that are easy to
misread. Adam looks worst at 50 steps, but that is a **budget artefact** — the same run
reaches 1.7e-22 by 500 steps; the cause is the β₁ momentum transient, verified by showing
that changing `lr` or `β₂` does not fix it. And AdamW beats Adam here only because **the
minimum of `f` sits at the origin, exactly where weight decay pulls** — move the minimum to
`(3, 1)` and AdamW stalls at a biased point while Adam converges. Both readings are traps
that a results table alone would walk straight into.

## Running it

```
jupyter notebook optimisers.ipynb
```

Run all cells; the final cell calls the three experiments in order and writes the figures
into a new `Optimiser_plots_<timestamp>/` directory. Requires `torch`, `numpy`,
`scikit-learn`, and `matplotlib`.
