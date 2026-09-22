# Grouped CV5 protocol

`CVOGL_DroneAerial` and `CVOGL_SVI` each use a deterministic grouped five-fold
development split.  The development set is the unchanged official train plus
val records; official test records are never input to fold generation.

Each grouping unit is a connected component of the query--satellite bipartite
graph.  Consequently no query or satellite image appears on both sides of any
fold's train/validation split.  Folds use seed 2024 and balanced greedy
assignment by component instance count.  All ten runs are independent
ImageNet-pretrained Swin-T Full Models with 25 epochs and natural RNG.

Training is validation-only.  Run `scripts/eval_fullmodel_grouped_cv5_all.sh`
only after all ten validation-selected best checkpoints exist; it evaluates
every fold on its own validation split and the unchanged official test split.
