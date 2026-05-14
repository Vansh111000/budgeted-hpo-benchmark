# Evaluation Budget Analysis Explanation

This analysis traces the Average Rank of various hyperparameter optimization methods across an increasing evaluation budget (`n_evals`). The findings highlight a critical trade-off between **early-stage exploitation** and **late-stage exploration**.

## The "Warmup Penalty" of CATS+ V2
At smaller evaluation budgets ($B=10$ to $B=100$), `catsplus_v2` frequently underperforms, floating at an average rank of ~3.5. This is expected behavior for an Estimation-of-Distribution approach. In its early stages, `catsplus_v2` focuses on randomly sampling the search space (warmup) to gather enough elite observations. During this phase, it is heavily disadvantaged compared to algorithms that aggressively exploit early good configurations.

## The Tipping Point
Between $B=100$ and $B=200$, `catsplus_v2` undergoes a phase transition. Once enough elite configurations are collected, it successfully maps out the optimal distribution of hyperparameters. Consequently, its performance dramatically spikes, jumping to a dominant average rank of 1.867 at $B=200$, and eventually stabilizing at an exceptional ~1.4 rank by $B=500$ and $B=1000$.

## The TPE Ceiling
Conversely, the Tree-structured Parzen Estimator (`tpe`) shines in constrained budgets ($B=20$ to $B=200$), maintaining a strong average rank of ~1.7. `tpe` is highly efficient at quickly navigating toward local optima. However, as the budget scales up to $B=500$ and $B=1000$, `tpe`'s rank degrades (falling to 2.6). This indicates that `tpe` struggles to escape early local optima, whereas the global distribution modeling of `catsplus_v2` allows it to discover better global configurations in the long run.

## Conclusion
If the evaluation budget is strictly limited (under 100 evaluations), `tpe` remains the most robust choice. However, for rigorous optimization tasks where a high evaluation budget ($B \ge 200$) is available, the distribution-learning mechanism of `catsplus_v2` drastically outperforms other search methods.
