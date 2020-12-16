+++
title = "How Deep Are scikit-learn's Histogram-based Gradient Boosted Trees?"
outputs = ["Reveal"]
+++

# How Deep Are scikit-learn's Histogram-based Gradient Boosted Trees?
# 🗻🚀🎄
Thomas J. Fan

{{< social >}}
{{< talk-link 2020-richmond-ds-meetup-gradient-boosting >}}

---

{{% section %}}

# Supervised Learning 📖

$$
y = f(X)
$$

- $X$ of shape `(n_samples, n_features)`
- $y$ of shape `(n_samples,)`

---

# Scikit-learn API 🛠

```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier()

clf.fit(X, y)

clf.predict(X)
```

{{% /section %}}

---

{{% section %}}

# HistGradient-*Boosting* 🚀

---

# Boosting 🚀

$$
f(X) = h_0(X) + h_1(X) + h_2(X) + ...
$$

$$
f(X) = \sum_i h_i(X)
$$

---

# Hist-*Gradient*-Boosting 🗻

---

# Gradient 🗻

{{% grid %}}

{{% g 1 %}}

## Regression

- `least_squares`
- `least_absolute_deviation`
- `poisson`

{{% /g %}}

{{% g 1 %}}

## Classificaiton

- `binary_crossentropy`
- `categorical_crossentropy`
- `auto`

{{% /g %}}

{{% /grid %}}

---

# Loss Function - `least_squares`

$$
L(y, f(X)) = \frac{1}{2}||y - f(X)||^2
$$

{{% grid %}}

{{% g 1 %}}

## Gradient

$$
\nabla L(y, f(X)) = -(y - f(X))
$$

{{% /g %}}

{{% g 1 %}}

## Hessian

$$
\nabla^2 L(y, f(X)) = 1
$$

{{% /g %}}
{{% /grid %}}

---

# Gradient Boosting 🗻🚀

- Initial Condition

$$
f_0(X) = C
$$

- Recursive Condition

$$
f_{m+1}(X) = f_{m}(X) - \eta \nabla L(y, f_{m}(X))
$$

where $\eta$ is the learning rate

---

# Gradient Boosting 🏂 - `least_squares`

- Plugging in gradient for least_square

$$
f_{m+1}(X) = f_{m}(X) + \eta(y - f_{m}(X))
$$

- Letting $h_{m}(X)=(y - f_{m}(X))$

$$
f_{m+1}(X) = f_{m}(X) + \eta h_{m}(X)
$$

- We need to learn $h_{m}(X)$!
- For the next example, let $\eta=1$

---

# Gradient Boosting 🏂 - (Example, part 1)

$$
f_0(X) = C
$$

{{< figure src="images/gb-p1.png" height="500px">}}

---

# Gradient Boosting 🏂 - (Example, part 2)

{{< figure src="images/gb-p2.png" height="500px">}}

---

# Gradient Boosting 🏂 - (Example, part 3)

{{< figure src="images/gb-p3.png" height="500px">}}

---

# Gradient Boosting 🏂 - (Example, part 4)

$$
f_{m+1}(X) = f_{m}(X) + h_{m}(X)
$$

{{< figure src="images/gb-p4.png" height="500px">}}

---

# Gradient Boosting 🏂 - (Example, part 5)

{{< figure src="images/gb-p5.png" height="500px">}}

---

# Gradient Boosting 🏂 - (Example, part 6)

{{< figure src="images/gb-p6.png" height="500px">}}

---

# Gradient Boosting 🏂 - (Example, part 7)

{{< figure src="images/gb-p7.png" height="500px">}}

---

# Gradient Boosting 🏂

With two iterations of boosting:

$$
f(X) = C + h_0(X) + h_1(X)
$$

## Prediction

For example, with $X=40$

$$
f(40) = 78 + h_0(40) + h_1(40)
$$

{{% /section %}}

---

{{% section %}}

# How to learn $h_m(X)$?

---

# 🎄🎄🎄🎄🎄🎄🎄
# 🎄🎄🎄🎄🎄🎄🎄
# 🎄🎄🎄🎄🎄🎄🎄
# 🎄🎄🎄🎄🎄🎄🎄
# 🎄🎄🎄🎄🎄🎄🎄

---

# Tree Growing 🌲

{{% grid %}}

{{% g 1 %}}

1. For every feature
    1. Sort feature
    1. For every split point
        1. **Evaluate split**
1. Pick best split

{{% /g %}}

{{% g 1 %}}

![](images/tree-growing-1.png)

{{% /g %}}

{{% /grid %}}

---

# How to evaluate split?
## `least_square`

- Recall Loss, Gradient, Hessian

$$
L(y, f(X)) = \frac{1}{2}||y - f(X)||^2
$$

$$
G = \nabla L(y, f(X)) = -(y - f(X))
$$

$$
H = \nabla^2 L(y, f(X)) = 1
$$

---

# How to evaluate split?
Maximize the Gain!

$$
Gain = \dfrac{1}{2}\left[\dfrac{G_L^2}{H_L+\lambda} + \dfrac{G_R^2}{H_R + \lambda} - \dfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]
$$

default $\lambda$: `l2_regularization=0`

---

# Tree Growing 🎄

{{% grid %}}

{{% g 1 %}}

## Are we done?

1. For every feature
    1. Sort feature
    1. For every split point
        1. Evaluate split
1. Pick best split

{{% /g %}}

{{% g 1 %}}

![](images/tree-growing-2.png)

{{% /g %}}

{{% /grid %}}

---

# Tree Growing 🎄

{{% grid %}}

{{% g 1 %}}

## Are we done?

1. For every feature
    1. Sort feature - _**O(nlog(n))**_
    1. For every split point - _**O(n)**_
        1. Evaluate split
1. Pick best split

{{% /g %}}

{{% g 1 %}}

![](images/tree-growing-2.png)

{{% /g %}}

{{% /grid %}}

{{% /section %}}

---

{{% section %}}

# *Hist*-GradientBoosting

---

# Binning! 🗑

{{% figure src="images/binning101.png" %}}

---

# Binning! 🗑

```py
# Original data
[-0.752,  2.7042,  1.3919,  0.5091, -2.0636,
 -2.064, -2.6514,  2.1977,  0.6007,  1.2487, ...]

# Binned data
[4, 9, 7, 6, 2, 1, 0, 8, 6, 7, ...]
```

---

# Histograms! 📊

{{% figure src="images/binned-gradient-hess.png" height="500px" %}}

---

# Histograms! 📊

{{% grid %}}

{{% g 1 %}}

## Overview

1. For every feature
    1. Build histogram _**O(n)**_
    1. For every split point - _**O(n\_bins)**_
        1. Evaluate split
1. Pick best split

{{% /g %}}

{{% g 1 %}}

![](images/gradient-split-points.png)

{{% /g %}}

{{% /grid %}}

---

# One More Trick 🎩

{{% figure src="images/histogram_subtraction.png" %}}

---

# Trees = $h_m(X)$ 🎄

$$
f(X) = C + \sum h_{m}(X)
$$

{{% /section %}}

---

{{% section %}}

# Overview of Algorithm 👀

1. Bin data
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. Find best splits
    1. Add tree to predictors
    1. Update gradients and hessians

---

# Implementation? 🤔

- Pure Python?
- Numpy?
- Cython?
- Cython + OpenMP!

---

# OpenMP! Bin data 🗑

1. _**Bin data**_
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. Find best splits by building histograms
    1. Add tree to predictors
    1. Update gradients and hessians

---

# OpenMP! Bin data 🗑


```python{|1}
for i in range(data.shape[0]):
    left, right = 0, binning_thresholds.shape[0]
    while left < right:
        middle = left + (right - left - 1) // 2
        if data[i] <= binning_thresholds[middle]:
            right = middle
        else:
            left = middle + 1
    binned[i] = left
```

---

# OpenMP! Bin data 🗑

```python{|2}
# sklearn/ensemble/_hist_gradient_boosting/_binning.pyx
for i in prange(data.shape[0], schedule='static', nogil=True):
    left, right = 0, binning_thresholds.shape[0]
    while left < right:
        middle = left + (right - left - 1) // 2
        if data[i] <= binning_thresholds[middle]:
            right = middle
        else:
            left = middle + 1
    binned[i] = left
```

---

# OpenMP! Building histograms 🌋

1. Bin data
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. Find best splits by _**building histograms**_
    1. Add tree to predictors
    1. Update gradients and hessians

---

# OpenMP! Building histograms 🌋

```python{|1-4|6-8}
# sklearn/ensemble/_hist_gradient_boosting/histogram.pyx
with nogil:
    for feature_idx in prange(n_features, schedule='static'):
        self._compute_histogram_brute_single_feature(...)

for feature_idx in prange(n_features, schedule='static',
                          nogil=True):
    _subtract_histograms(feature_idx, ...)
```

---

# OpenMP! Find best splits ✂️

1. Bin data
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. _**Find best splits**_ by building histograms
    1. Add tree to predictors
    1. Update gradients and hessians

---

# OpenMP! Find best splits ✂️

```py
# sklearn/ensemble/_hist_gradient_boosting/splitting.pyx
for feature_idx in prange(n_features, schedule='static'):
    # For each feature, find best bin to split on
```

---

# OpenMP! Splitting ✂️

```py
# sklearn/ensemble/_hist_gradient_boosting/splitting.pyx
for thread_idx in prange(n_threads, schedule='static',
                         chunksize=1):
    # splits a partition of node
```

---

# OpenMP! Update gradients and hessians 🏔

1. Bin data
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. Find best splits by building histograms
    1. Add tree to predictors
    2. _**Update gradients and hessians**_

---

# OpenMP! Update gradients and hessians 🏔

`least_squares`

```py
# sklearn/ensemble/_hist_gradient_boosting/_loss.pyx
for i in prange(n_samples, schedule='static', nogil=True):
    gradients[i] = raw_predictions[i] - y_true[i]
```

{{% /section %}}

---

{{% section %}}

# Hyper-parameters 📓

---

# Hyper-parameters: Bin Data 🗑

1. _**Bin data**_
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. Find best splits by building _**histograms**_
    1. Add tree to predictors
    1. Update gradients and hessians

---

# Hyper-parameters: Bin Data 🗑

`max_bins=255`

{{% figure src="images/binning101.png" height="500px"%}}

---

# Hyper-parameters: Loss 📉

1. Bin data
1. _**Make initial predictions (constant)**_
1. Calculate _**gradients and hessians**_
1. Grow Trees For Boosting
    1. Find best splits by building histograms
    1. Add tree to predictors
    1. _**Update gradients and hessians**_

---

# Hyper-parameters: Loss 📉

- `HistGradientBoostingRegressor`
    - `loss=least_squares` (default)
    - `least_absolute_deviation`
    - `poisson`

- `HistGradientBoostingClassifier`
    - `loss=auto` (default)
    - `binary_crossentropy`
    - `categorical_crossentropy`

- `l2_regularization=0`

---

# Hyper-parameters: Boosting 🏂

1. Bin data
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For _**Boosting**_
    1. Find best splits by building histograms
    1. Add tree to predictors
    1. Update gradients and hessians

---

# Hyper-parameters: Boosting 🏂

- `learning_rate=0.1` ($\eta$)
- `max_iter=100`

$$
f(X) = C + \eta\sum_{m}^{\text{max_iter}}h_{m}(X)
$$

---

# Hyper-parameters: Boosting 🏂

{{< figure src="images/boosting_p1.png" height="600px">}}

---

# Hyper-parameters: Boosting 🏂

{{< figure src="images/learning_rate_p1.png" height="600px">}}

---

# Hyper-parameters: Grow Trees 🎄

1. Bin data
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. _**Grow Trees**_ For Boosting
    1. Find best splits by building histograms
    1. Add tree to predictors
    1. Update gradients and hessians

---

# Hyper-parameters: Grow Trees 🎄

- `max_leaf_nodes=31`
- `max_depth=None`
- `min_samples_leaf=20`

---

# Hyper-parameters: Grow Trees 🎄

{{< figure src="images/max_leaf_nodes_p1.png" height="600px">}}

---

# Hyper-parameters: Grow Trees 🎄

{{< figure src="images/max_depth_p1.png" height="600px">}}

---

# Hyper-parameters: Early Stopping 🛑

1. Bin data
1. _**Split into a validation dataset**_
1. Make initial predictions (constant)
1. Calculate gradients and hessians
1. Grow Trees For Boosting
    1. ...
    1. _**Stop if early stop condition is true**_

---

# Hyper-parameters: Early Stopping 🛑

- `early_stopping='auto'` (enabled if `n_samples>10_000`)
- `scoring='loss'`
- `validation_fraction=0.1`
- `n_iter_no_change=10`
- `tol=1e-7`

---

# Hyper-parameters: Early Stopping 🛑

{{< figure src="images/early_stopping_p1.png" height="600px">}}

---

# Hyper-parameters: Misc 🎁

- `verbose=0`
- `random_state=None`
- `export OMP_NUM_THREADS=8`

{{% /section %}}

---

{{% section %}}

# Recently Added Features
- Missing values (0.22)
- Monotonic constraints (0.23)
- Poisson loss (0.23)
- Categorical features (0.24)

---

# Missing Values (0.22)

```python
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
gbdt.predict(X)
# [0 0 1 1]
```

---

# Monotonic Constraints (0.23)

```python
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

X, y = ...

gbdt_no_cst = HistGradientBoostingRegressor().fit(X, y)
gbdt_cst = HistGradientBoostingRegressor(monotonic_cst=[1, 0]).fit(X, y)
```

---

# Monotonic Constraints (0.23)

```python
from sklearn.inspection import plot_partial_dependence

disp = plot_partial_dependence(
    gbdt_no_cst, X, features=[0], feature_names=['feature 0'], line_kw={...})
plot_partial_dependence(gbdt_cst, X, features=[0], line_kw={...}, ax=disp.axes_)
```

{{< figure src="images/monotonic_cst.png" height="450px">}}

---

# Poisson Loss (0.23)

```python
hist_poisson = HistGradientBoostingRegressor(loss='poisson')
```

{{< figure src="images/poisson_hist.png" height="450px">}}

---

# Categorical Features (0.24)

From [categorical example](https://scikit-learn.org/dev/auto_examples/ensemble/plot_gradient_boosting_categorical.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-categorical-py)

```python
categorical_mask = ([True] * n_categorical_features +
                    [False] * n_numerical_features)
hist = HistGradientBoostingRegressor(categorical_features=categorical_mask)
```

{{< figure src="images/categorical_features.png" height="450px">}}

{{% /section %}}

{{% section %}}

# Compared to Other Libraries

---

# XGBoost

```bash
conda install -c conda-forge xgboost
```

```python
from xgboost import XGBClassifier
xgb = XGBClassifier()
```

- GPU training
- Networked parallel training
- Sparse data

---

# LightGBM

```bash
conda install -c conda-forge lightgbm
```

```python
from lightgbm.sklearn import LGBMClassifier
lgbm = LGBMClassifier()
```

- GPU training
- Networked parallel training
- Sparse data

---

# CatBoost

```bash
conda install -c conda-forge catboost
```

```python
from catboost.sklearn import CatBoostClassifier
catb = CatBoostClassifier()
```

- Focus on categorical features
- Bagged and smoothed target encoding for categorical features
- Symmetric trees
- GPU training
- Tooling

---

# Benchmark 🚀
## HIGGS Boson

- 8800000 samples
- 28 features
- binary classification (1 for signal, 0 for background)

---

## Current Benchmark Results

{{% grid %}}

{{% g 1 %}}

| library  | time | roc auc | accuracy |
|----------|------|---------|----------|
| sklearn  | 66s  | 0.8126  | 0.7325   |
| lightgbm | 42s  | 0.8125  | 0.7323   |
| xgboost  | 45s  | 0.8124  | 0.7325   |
| catboost | 90s  | 0.8008  | 0.7223   |

{{% /g %}}

{{% g 1 %}}

### Versions

- `xgboost=1.3.0.post0`
- `lightgbm=3.1.1`
- `catboost=0.24.3`

{{% /g %}}
{{% /grid %}}

{{% /section %}}

---

# Conclusion

{{% grid middle %}}

{{% g 1 %}}

## Future Work
- Sparse Data
- Improve performance when compared to other frameworks.
- Better way to pass feature-aligned metadata to estimators in a pipeline.

Learn more about [Histogram-Based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)

```bash
pip install scikit-learn==0.24.0rc1
```

{{% /g %}}

{{% g 1 %}}

Thomas J. Fan
{{< social >}}
{{< talk-link 2020-richmond-ds-meetup-gradient-boosting >}}

{{% /g %}}
{{% /grid %}}
