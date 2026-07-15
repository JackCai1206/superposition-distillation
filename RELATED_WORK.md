# Related Work

Curated — only papers we've actually read/reviewed go here. Each entry: what it shows,
and its precise relation to this project (superposition / perturbed-input distillation for LMs).

---

## Srinivas & Fleuret, *Knowledge Transfer with Jacobian Matching* (ICML 2018) — [arxiv 1803.00443](https://arxiv.org/abs/1803.00443)

**Core result (Prop. 1).** For squared-error distillation with Gaussian input noise `ξ=σz`:
```
E_ξ[ Σ_i (T^i(x+ξ) − S^i(x+ξ))^2 ] = Σ_i (T^i(x)−S^i(x))^2  +  σ^2 Σ_i ‖∇_x T^i(x) − ∇_x S^i(x)‖^2  +  O(σ^4)
```
i.e. **distilling on noise-perturbed inputs ⟺ ordinary distillation + (σ²·) matching teacher/student input-Jacobians** (`∇_x` = gradient of an output w.r.t. the input). For ReLU nets the O(σ⁴) vanishes exactly within a linear region. They train with *explicit* analytic Jacobians (double-backprop; match only the correct-class output for cost), using noise as the justification. Setting: CIFAR-100 / MIT-Scenes image classification. Headline: the Jacobian term helps **most in the low-data regime** (~⅕ the data for full-data accuracy) and the gain **vanishes at full data** — fundamentally a *sample-efficiency* method.

**Relation to us — this is the theoretical backbone, and our contribution is the LM port.** Our "perturbed-input KD" is exactly their Prop. 1; our σ-sweep traces out their `σ²` term. What's new for LMs is precisely where their assumptions break:
- **Discrete inputs** → no `∇_x` over tokens; we perturb in embedding / **one-hot vocab** space.
- **Their theorem assumes teacher & student share the input space (`same D`).** LM distillation is **different-width** (2048 vs 576 embeddings) → embedding-space Jacobians aren't comparable. Perturbing the **shared one-hot `o∈ℝ^V`** recovers a common input space so `∇_o T`, `∇_o S` live in the same `ℝ^V` and *can* be matched — making their result well-defined across widths, a case they cannot handle.
- **Explicit Jacobians are infeasible at LM `T×V×d` scale** → the noise route (which they proved equivalent) is the *only* practical vehicle, not a shortcut.
- **Regime flip:** their motivation is few labels; LM pretraining is compute-bound / ~infinite-data → question becomes *signal-per-FLOP*, and whether the term still helps when data is free (they say it vanishes at full data on images — open at LM scale).
- **New knob:** a **coherence bound on σ** (perturb the one-hot too far → teacher's own target degrades off-manifold), tied to our superposition/packing coherence story — absent from their paper.

## Czarnecki et al., *Sobolev Training for Neural Networks* (NeurIPS 2017)

Train a network to match a target function's **values and derivatives** (Sobolev norm) — matching slopes as well as points pins the function with far fewer samples (Hermite-interpolation intuition). The analytic-derivative sibling of the above.

**Relation to us.** Motivates matching the teacher's *gradients*, not just outputs. We take the noise route (Srinivas & Fleuret) rather than explicit derivative-matching, because full Jacobians are intractable at LM scale; and we operate in the compute-bound regime where Sobolev's sample-efficiency payoff is *not* the point.
