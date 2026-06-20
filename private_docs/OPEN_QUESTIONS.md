# Open Questions

Updated: 2026-06-20

- Should the final range use the training 99.9th percentile with clipped tail,
  another robust quantile, or a fixed biological/acquisition limit?
- Should `FITC_Sum` use linear or log1p-spaced bins, and how many bins are
  stable across seeds?
- Which small latent/hidden dimensions avoid overcapacity for 94 training
  groups while retaining useful sample variation?
- Which beta or beta schedule balances reconstruction and active latent
  coordinates?
- Which downstream sample-level task should select the final pretrained model?
- Does `probability_mass` outperform legacy `density` under identical splits,
  seeds, and capacity?
