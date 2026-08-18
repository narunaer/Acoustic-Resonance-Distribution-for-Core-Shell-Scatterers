# Data layout

`density/final` contains the density dataset used directly in the article.

For `mixed` and `velocity`:

- `raw` contains the untouched output of the automatic resonance detector;
- `manual_corrected` contains the explicit hand-curated final dataset used in the article.

The raw and corrected stages are deliberately kept in separate directories for auditability.

The same staging is used under both `data/checkpoints/` and `data/exports/`.
