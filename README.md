### Collaborative Online Sample Selection (COSS)

The main purpose of COSS is to select a subset of samples for human labeling. Given the limited labeling effort, this subset should be able to represent the entire dataset in measuring service accuracy with the least possible error.

Here are some main concepts in this approach:

1. **PSI** represents how significant a sample is towards service accuracy measurement. The signifance of a sample is refers to how important it is and how uncertain its label distribution is. PSI is defined as:
```math
PSI = CERT \times Stability Gap
```

2. **CERT** represents the importance of a sample. It is implemented using Spearman correlation ranking. The more important a sample is, the higher the CERT value. CERT is defined as:
```math
CERT = 1 - \frac{6 \Sigma d_i^2}{n(n^2 - 1)} = 1 - \rho
```

3. **Stability Gap** represents the certainty of label distribution of a sample. It is implemented using Bayes' Theorem. The higher the certainty of a label distribution, the lower the Stability Gap. Stability Gap is defined as:
```math
Stability Index = abs\left(\frac{B(x'+2, y'+1)}{B(x'+1, y'+1)} - \frac{B(x+2, y+1)}{B(x+1, y+1)}\right)
Threshold = (1 - x) e^{-bx}
Stability Gap = max\left((Stability Index - Threshold), 0\right)
```