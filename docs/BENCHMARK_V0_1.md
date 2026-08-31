# Benchmark v0.1 Results

Run date: 2026-08-31  
Configuration: `configs/burgers_v0_1.json`  
Claim level: **controlled instrument validation only**

## Best discrepancy by execution budget

| Policy | 20 | 40 | 80 | 120 | 220 |
|---|---:|---:|---:|---:|---:|
| Adaptive v0.1 | **1.181113** | **1.243190** | **1.243190** | **1.243190** | **1.243190** |
| Fixed enumeration | 0.861920 | 1.052578 | 1.052578 | 1.052578 | **1.243190** |
| Random seed 11 | **1.191600** | 1.191600 | 1.191600 | 1.191600 | 1.191600 |
| Random seed 23 | 0.800979 | 0.917446 | 0.917446 | 0.917446 | 1.191600 |
| Random seed 47 | 1.052578 | 1.052578 | 1.185074 | 1.185074 | 1.185074 |

Bold values mark the strongest result at a budget, including near-equal or identical maxima where
appropriate. The scalar score is an acquisition summary, not a complete physical conclusion.

## Main observation

Adaptive v0.1 discovered probe 56, the strongest witness found by any policy, within 40 operator
comparisons. Fixed enumeration required 220 comparisons to reach the same probe. None of the three
random trials found it within 220 comparisons, although random seed 11 found a slightly stronger
result than adaptive at budget 20.

This is evidence that the adaptive instrument can exploit cheap structural descriptors on this
constructed problem. It is not yet evidence that it generalizes.

## Local persistence

The best adaptive witness was perturbed 32 times in amplitude and phase space.

| Measure | Result |
|---|---:|
| Source score | 1.243190 |
| Neighborhood median | 1.240538 |
| Neighborhood minimum | 1.157329 |
| Neighborhood maximum | 1.300516 |
| Fraction retaining at least 80% | **1.00** |

The discovered witness therefore marks a local discrepancy region rather than an isolated numerical
point under this perturbation test.

## Honest interpretation

The candidate deliberately removes high Fourier modes, while the policy receives descriptors that
include spectral structure. The policy is blind to the implementation but the benchmark is designed
around a discoverable relationship. The experiment establishes that the plumbing, artifacts,
budgets, comparison policies, and neighborhood test work together coherently.

The next decisive test is to freeze this policy and replace the constructed candidate with a trained
FNO that was not designed to produce the same failure structure.

