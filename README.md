# Fast and More Powerful Selective Inference for Sparse High-order Interaction Model

This package implements a "homotopy mining" method by exploiting the best of both homotopy and (pattern) mining methods for conditional Selective Inference of the Sparse High-order Interaction Model (SHIM).


## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org)
- [statsmodels](https://www.statsmodels.org/)
- [pandas](https://pandas.pydata.org)

We recommend installing or updating anaconda to the latest version and use Python 3 (We used Python 3.8.5).

All commands are run from the terminal.

## Reproducibility

**NOTE**: Due to the randomness of data generating process, we note that the results might be slightly different from the paper. However, the overall results for interpretation will not change.

All the figure results are saved in folder "/results"


- False Positive Rate (Figure 2a)
	```
	>> python ex1_fpr_synthetic_data.py
	```

- True Positive Rate (Figure 2b)
	```
	>> python ex1_tpr_synthetic_data.py
	```
 
- Confidence Interval (Figure 2c)
	```
	>> python ex1_ci_synthetic_data.py
	```
 
- Checking the uniformity of the pivot
	```
	>> python ex1_pivot_synthetic_data.py
	```


