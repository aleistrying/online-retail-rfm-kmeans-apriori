# Data

This project uses the **UCI Online Retail Data Set** — 541,909 transactions from a UK-based
online retailer, December 2010 to December 2011.

- Dataset page (licence + description): https://archive.ics.uci.edu/dataset/352/online+retail
- Direct file: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

The raw file is not committed here (it's ~23 MB, and the derived cleaned parquet is a further
~4.4 MB — both over what's worth keeping in git history for a coursework repo).

## To reproduce

1. Download `Online Retail.xlsx` from the link above.
2. Rename it to `online_retail.xlsx` and place it in `notebook/`, next to
   `online-retail-mining.ipynb` (the notebook reads it from `./online_retail.xlsx`).
3. Run the notebook top-to-bottom.

## Licence

The UCI Online Retail dataset is provided by the UCI Machine Learning Repository for research/
academic use — see the dataset page for the exact terms. It keeps its own licence; the MIT
licence in this repo covers only my analysis code.
