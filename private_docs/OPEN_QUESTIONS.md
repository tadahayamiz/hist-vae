# Open Questions

Updated: 2026-06-19

- What is the intended prediction or representation-learning objective for the
  `label` column?
- Should train/test separation occur strictly by `sample_name`, and are there
  higher-level batch or acquisition identifiers that must also be separated?
- Should `max_vals` use a fixed domain limit, the training maximum, or a robust
  training quantile with explicit overflow handling?
- How many bins should be used for `FITC_Sum`, and should values be transformed
  before histogramming?
- Does density improve downstream stability compared with count mode under the
  same split, seed, and model configuration?
