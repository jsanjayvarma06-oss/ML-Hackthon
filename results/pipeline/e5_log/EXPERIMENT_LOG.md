# E5: junction / direction errors — HMM experiments (train GroupKFold CV only)

Thread "Fix junction and direction errors". student_test was never touched (not even linked into the working copy).
All numbers: pooled Top-1 over the 185,107 labelled train points, classifier probabilities = out-of-fold (GroupKFold(5) by trip, tuned LightGBM params from results/models/best_params.json). Unlabelled points have no OOF probability and get 0.5 (same as scripts/hmm_eval.py).
Base = branch claude/project-thread-0jgx9i @ 90adec6 (K=8). Scripts: experiment_scripts/ (they cache candidates + network distances per trip, then re-decode fast).

## Result in one line
K=8: 0.9388 -> 0.9488 (patch). K=16 (lead's candidate change, own re-run): 0.9407 base-decoder -> **0.9509** with the patch (best overall 0.9510). Honest held-out estimate of the parameter choice: 0.9480 / 0.9505 (see E5f). So we are AT the 95% line, not clearly above it.

## What the errors really are (E5a, baseline, K=8)
Breakdown of the 6.1% wrong points: adjacent arc 2.18%, candidate miss 0.92%, other 1.11%, reverse twin 1.92% (1.56% of it at speed >= 2 m/s, so not just "stopped").
Inspecting the biggest error runs (trip 84, 10, 33, 2): most are LONG STATIONARY stretches (car parked, GPS jitter 5-10 m, "speed" 3-5 m/s is just jitter), 100-340 points each, where two directions of one road (or 2-3 arcs at a junction) are equally close and the classifier says ~0.5/0.5. One coin flip then decides 100-300 points. Two concrete causes found:
1. **Time gaps reset the HMM.** clean_track starts a new segment after a gap > 60 s. A parked car that reports again after a 70 s pause (trip 84, row 13222) restarts Viterbi with no memory, the first point picks a direction at random and the whole stretch inherits it.
2. **Switching to the reverse twin was almost free** whenever the point is near an arc end (network distance of the U-turn = 2*(1-t)*L is small), so the decoder happily flipped direction on jitter and then stayed there.

## Experiments (all train CV; K=8 unless noted)
| id | change | Top-1 | verdict |
|---|---|---|---|
| E5a | baseline (beta 10, floor 25, w 0.1) | 0.9388 | reproduces E3 |
| E5b | forward-backward posterior decoding (max marginal instead of Viterbi), beta 5-10, w 0.1-1.0 | 0.9139-0.9323 | worse, rejected (run stopped after 7 settings) |
| E5c | bridge time-gap segments (link last point before a gap to first after it, any gap distance; capped at 5 km in the patch) | 0.9409 | +0.21, kept |
| E5d | + U-turn penalty u nats on arc -> its reverse twin (beta 10, floor 25, w 0.1): u=2 / 5 / 10 / 20 / 40 | 0.9460 / 0.9460 / 0.9457 / 0.9463 / 0.9467 | +0.5, kept (any u >= 2 works; 100 used) |
| E5e | grid beta {5,10,20}, floor {25,50}, w {0.05..0.5}, u {20,40,100} (72 runs) then beta {30,50,100}, floor {100,200} | best beta 20 floor 50 w 0.1 u 100 = 0.9476; beta 30 = **0.9488**; beta 50 = 0.9446; beta 100 = 0.9412; floor 100 / 200 = 0.9476 / 0.9471 | beta 30, floor 50 |
| E5g | constant arc-switch penalty sw (any a != b) 0.5 / 1 / 2 / 4 | 0.9467 / 0.9367 / 0.9102 / 0.8694 | worse, rejected |
| E5h | distance mixed into emission (wd 0.01-0.1) | 0.9476-0.9479 | no real gain, rejected |
| E5i | speed-scaled beta (beta + b1 * d_gps), b1 0.1-1.0 | 0.9460-0.9488 | no gain, rejected |
| E5j | new classifier features: mean distance of the same arc to the +-1..3, +-5..15, +-25..100 neighbouring points (+ "relative to best candidate" versions), LightGBM retrained OOF | classifier-only 0.8509 -> 0.8619 (+1.1) but classifier+HMM 0.9428-0.9437 (best params) | **worse after the HMM** (-0.5); rejected. Interesting: a better classifier alone did not help the decoder. |
| E5k | K=16 candidates (own copy, K=16 dataset + retrained LightGBM OOF; classifier-only 0.8504) with the patch | beta 15, w 0.07, floor 50, u 100 = **0.9509**; beta 20, w 0.05 = 0.9507; beta 30, w 0.03 = 0.9502; beta 10, w 0.1 = 0.9510; baseline decoder (beta 10 floor 25 w 0.1, gap bridging) = 0.9407 | the combination crosses 0.95 |

## E5f: how optimistic is the parameter choice?
Grid beta {10,15,20,30,40} x w {0.03,0.05,0.07,0.1} (floor 50, u 100) on K=16, trips split in two halves (even / odd index). Pick best on one half, score on the other:
- chosen on A (beta 40, w 0.03): A 0.9514, held-out B **0.9480**
- chosen on B (beta 10, w 0.1): B 0.9515, held-out A **0.9505**
Whole grid range 0.9407-0.9510; 9 of 20 settings are >= 0.949. Honest expectation on unseen trips: about 94.8-95.1%. Do not promise "95%+" in the report; report "about 95% on train CV" with this caveat.

## Recommended parameters (K=16, classifier + HMM)
beta 15, floor 50, emission_weight 0.07, uturn 100 (a point inside a plateau, 0.9509; neighbours 0.9491-0.9507). Re-fit them with `scripts/hmm_eval.py` after the lead's own K=16 OOF exists (that file only grids beta {1..50}, floor {25..200}, weight {0.05..1}; add 0.03/0.07 to the weights and uturn is read from DEFAULT_PARAMS = 100).

## What is NOT fixed (remaining errors with K=8 + patch: 5.1%)
adjacent arc 2.05%, candidate miss 0.92% (K=16 halves it), other 1.03%, reverse twin 1.12%. They sit in long stationary stretches at junctions, where the label comes from the route-alignment tie-break (a monotone alignment decides which of two equally close arcs a parked car "belongs" to). Without the route that is not predictable from the track alone. A fix would have to model that tie-break (e.g. label stationary clusters with the arc the car leaves onto) — not tried.

## Caveats
- Same-data tuning: parameters picked on the train trips they are scored on (E5f quantifies it).
- OOF probabilities exist only for labelled rows; unlabelled points use 0.5 here, but the live pipeline uses real model probabilities. The patch does not depend on this, but expect small differences.
- Gap bridging merges segments across ANY gap up to 5 km (trip 90's 42 h gap and trip 78's 52,810 s gap included); decisions are unchanged beyond 5 km because every pair is unreachable and gets the same floor.
- Not run: student_test, the live API, the full test/lint.
