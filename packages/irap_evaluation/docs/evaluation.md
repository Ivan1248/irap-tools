# Evaluation definitions

This document defines what `irap_evaluation` computes when it scores a prediction file ([format](prediction_format.md)), and ends with how to [interpret the results](#interpreting-results). The definitions match the training-time evaluation of `vidlu_irap_gaim` (`get_irap_metrics`): on its model set, a model gets the scores reported during training, up to float32 rounding (`tests/test_vidlu_parity.py`).

Scoring a file on an evaluation set (`evaluate_predictions`) has these steps:

1. Select the segments and attributes ([evaluation sets](#evaluation-sets), [attributes](#attributes)).
2. Match the predictions to the labels, and score the invalid cells according to the null policy ([matching](#matching), [invalid cells](#invalid-cells)).
3. Accumulate statistics per road sequence ([statistics](#statistics)).
4. Compute the metrics from the summed statistics ([metrics](#metrics)).

The runs of a method are then combined by their mean ([methods](#methods)). `compute_bootstrap_intervals` and `compare_methods` resample the road sequences and add the spread of a method's runs ([confidence intervals](#confidence-intervals), [comparing methods](#comparing-methods)).

## Terms

- **Method** and **run**: a file holds the predictions of one run of a method, e.g. one training seed. Files with the same `method.name` are runs of one method and are told apart by `method.seed` ([format](prediction_format.md#method)). A method with a single run needs no seed.
- **Road sequence**: the segments of one road in driving order, one entry of `road_id_to_segment_id_sequence.json`.
- **Class index**: the position of a value in `attribute_value_to_irap_number` of `attribute_metadata.json`. It need not follow iRAP code order. Class 0 is the first value.
- **Cell**: the predicted distribution of one attribute for one segment. It is invalid if the model gave no usable prediction (null in the file, see [Invalid cells](#invalid-cells)).

## Evaluation sets

For context offsets $o_1, \ldots, o_m$, the evaluation set of a split holds its segments that pass all of these filters (`irap_data.select_split_segments`):

1. **Image**: the segment has an entry in `segment_id_to_data_paths_rel.json`.
2. **Labels**: each iRAP code is mapped to its class index. A code is missing if it is absent, None, `'None'`, negative, or not a class of the attribute. Vietnam leaves the attribute unlabeled for that segment. BH drops the segment if any attribute is missing. Any other value, e.g. a float, is an error.
3. **Context**: for a segment at position $p$ of a sequence of length $L$, every $p + o_j$ lies in $[0, L)$, and the segment there has an image. Context segments need no label.
4. **N-context (BH only)**: the segment and the 10 segments on each side of it in its sequence are all in the same `seg_to_res/<split>.pickle` (train, val or test).

A file is scored on up to two sets:

- **Reference set**: offsets −4 … 0 for Vietnam and −10 … 10 for BH (`DATASET_PRESETS`). It is the same for every model, so its scores are comparable across models.
- **Model set**: the model's own offsets, from the file header or `--context-offsets`. This is the set of the training-time evaluation. If the offsets lie within the reference window, the model set contains the reference set.

`select_evaluation_sets` (used by `irap-eval evaluate`) selects the sets as follows:

- A file that lacks segments of its model set is incomplete or of another metadata build, and is refused.
- A model whose offsets reach beyond the reference window lacks some reference segments. It is scored only on its model set, with a note, or refused if its offsets are unknown.
- If the two sets are equal, the model is scored only on the reference set.

## Attributes

The scored attributes are the canonical iRAP subset (`irap_data.attrs.get_attrs_to_include`, 41 attributes), in its order, minus those without a label in the evaluation set. On Vietnam validation, 34 remain. `--attributes all` starts from every attribute of the dataset. `--only-predicted` also drops the attributes the file does not predict, which are otherwise an error.

## Matching

- Every segment of the set must have a prediction. Extra predictions are ignored.
- Classes are matched by iRAP code. The file must predict exactly the dataset's codes of each attribute, in any order.
- The predicted class of a cell is the argmax of its distribution. On ties, it is the first maximum in dataset class order.

## Invalid cells

An invalid cell is a null in the file: the method gave no usable prediction of the attribute for the segment. The producer of the file decides what is usable. For VLMs, `vidlu_irap_gaim` (`is_usable_prediction`) marks a cell invalid if the response leaves out the attribute or names a value that matches none of its classes. Both cases are one category, since the boundary between them depends on the response parser. An invalid cell is not the same as:

- an unlabeled segment, which has no label and is not scored for the attribute,
- a missing row, i.e. a segment without a prediction, which makes the file incomplete (see [Evaluation sets](#evaluation-sets)).

The null policy (`NullPolicy`) selects how invalid cells are scored:

- **`first_class`** (default): in metrics computed from predicted classes, the cell counts as a prediction of class 0, as in the training-time evaluation. In NLL and Brier, it counts as the uniform distribution, which gives $\ln K$ and $1 - 1/K$ for $K$ classes (a one-hot distribution of class 0 would give an infinite NLL for any other label).
- **`exclude`**: the cell is not scored, as if unlabeled. Each run leaves out its own invalid cells, so runs with invalid cells are scored on different segments.

Means over runs and comparisons therefore need `first_class`. `irap-eval evaluate` scores the runs with `--null-policy` (default `first_class`) and writes the means over the runs of a method (`method_averages_*.csv`) only with `first_class`. `irap-eval compare` always uses `first_class`.

The tables and documents have a `null_policy` column or field. In the library, `compute_method_metrics` and `compute_bootstrap_intervals` refuse several runs scored with `exclude` if any of them has invalid cells, and `compare_methods` refuses runs scored with `exclude` if any of them has invalid cells.

## Statistics

For each attribute and each road sequence $g$, the segments with a label for the attribute give:

- The confusion matrix $C^{(g)}_{ij}$: the number of segments with label $i$ and predicted class $j$.
- For `probs` files, the per-label sums
  $$s^{(g)}_{\mathrm{NLL},i} = \sum_{t:\,y_t=i} -\ln q_{t,y_t}, \qquad s^{(g)}_{\mathrm{Brier},i} = \sum_{t:\,y_t=i} \sum_k \left(q_{t,k} - [k = y_t]\right)^2,$$
  where $y_t$ is the label of segment $t$ and $q_t$ is its predicted distribution.

The metrics of any set of sequences are computed from the summed statistics $C = \sum_g C^{(g)}$ and $s = \sum_g s^{(g)}$. `EvaluationResult.statistics` keeps the per-sequence statistics, from which `compute_grouped_metrics` recomputes any metric without the predictions.

## Metrics

The metrics of an attribute are computed from $C$, with, for each class $k$:
- $\mathrm{TP}_k = C_{kk}$,
- the support $a_k = \sum_j C_{kj}$,
- the number predicted, $\hat a_k = \sum_i C_{ik}$,
- $n = \sum_k a_k$.

| Name | Definition |
|---|---|
| `P`, `R` | $P_k = \mathrm{TP}_k / \hat a_k$, $\;R_k = \mathrm{TP}_k / a_k$ (0 if the denominator is 0) |
| `F1` | $F_k = 2 P_k R_k / (P_k + R_k)$ (0 if $P_k + R_k = 0$) |
| `IoU` | $\mathrm{TP}_k / (a_k + \hat a_k - \mathrm{TP}_k)$ (0 if the denominator is 0) |
| `mP`, `mR`, `mF1`, `mIoU` | the mean over the classes $\mathcal{K} = \{k : a_k > 0 \text{ or } \hat a_k > 0\}$ |
| `A` | $\sum_k \mathrm{TP}_k / n$ |
| `n` | $n$ |
| `MCC` | $\dfrac{n \sum_k \mathrm{TP}_k - \sum_k \hat a_k a_k}{\sqrt{(n^2 - \sum_k \hat a_k^2)(n^2 - \sum_k a_k^2)}}$ (NaN if all labels or all predictions are one class) |
| `kappa` | $\dfrac{n \sum_k \mathrm{TP}_k - \sum_k \hat a_k a_k}{n^2 - \sum_k \hat a_k a_k}$ |
| `NLL`, `Brier` | $\sum_k s_k / n$ |
| `cNLL`, `cBrier` | $s_k / a_k$ per class (NaN if $a_k = 0$) |
| `mNLL`, `mBrier` | the mean of `cNLL`, `cBrier` over the classes with $a_k > 0$ |

A class in $\mathcal{K}$ that is labeled but never predicted, or predicted but never labeled, has $F_k = 0$. Other undefined values (0/0) are NaN. `NLL` is infinite if a valid cell gives its label probability 0. The probabilistic metrics are not defined for `hard` files, and requesting them is an error.

**Support threshold.** The suffix `_suppN` restricts a mean over classes to $\mathcal{K}_N = \{k : a_k \ge N\}$, e.g. `mF1_supp10`. `nc_suppN` is $|\mathcal{K}_N|$.

**Attribute average.** The prefix `a` averages a metric over the scored attributes where it is not NaN, e.g. `amF1`. Infinite values are included. Each attribute counts equally, so `aNLL` is the mean of the per-attribute NLLs. Per-class metrics are not averaged.

The iRAP protocol (`get_irap_metric_names`) reports the following. Its main metric is `amF1`.

- **Attribute averages**: `amF1`, `amP`, `amR`, `aMCC`, `aA`, `amF1_supp5` and `amF1_supp10`. `probs` files also get `aNLL`, `aBrier`, `amNLL`, `amNLL_supp5` and `amNLL_supp10`.
- **Per attribute**: `mF1`, `A`, `n`, `MCC`, `mF1_suppN` and `nc_suppN`. `probs` files also get `NLL`, `Brier` and `mNLL`.

Results keep only the attribute averages and per-attribute metrics. `compute_class_metrics(result)` computes the per-class metrics (`P`, `R`, `F1`, `IoU`, `cNLL`, `cBrier`), support and confusion matrix of each attribute from its statistics.

## Methods

The value of a metric for a method with runs $i = 1, \ldots, R$ on the same evaluation set is the mean $\bar m = \frac{1}{R} \sum_i m_i$ of the runs' values (`compute_method_metrics`). The runs must have the same segments, sequences, attributes, metrics and null policy. A method with a single run has the values of that run.

## Confidence intervals

The intervals cover two sources of variation (`--bootstrap B`, `--seed`, default 0):

- **Road sequences.** Segments of a road sequence are correlated, so the bootstrap resamples sequences, not segments.
- **Runs.** Runs of a method differ, e.g. by training seed. Their mean is uncertain, and more so with few runs.

`compute_bootstrap_intervals(runs)` computes them as follows:

1. Each of the $B$ resamples draws the $G$ sequences of the set $G$ times with replacement. This gives multiplicities $w \sim \mathrm{Multinomial}(G; 1/G, \ldots, 1/G)$. All runs share the resamples.
2. All metrics of each run, including the means and averages, are recomputed from $\sum_g w_g C^{(g)}$ and $\sum_g w_g s^{(g)}$, giving $m^*_i$. A sequence with weight 0 adds nothing, also if its NLL sum is infinite.
3. The resampled value of the method is
   $$v^* = \bar m^* + \frac{\mathrm{sd}(m^*_1, \ldots, m^*_R)}{\sqrt R}\, t^*, \qquad t^* \sim t_{R-1},$$
   with the sample standard deviation and one Student-t draw per resample. This is the posterior of the mean of normally distributed run values under the prior $p(\mu, \sigma) \propto 1/\sigma$. With one run, $v^* = m^*_1$, and the interval covers only the road sequences. If $\bar m^*$ is infinite, $v^*$ is $\bar m^*$.
4. The interval at confidence $c$ (default 0.95) runs from the $(1-c)/2$ to the $(1+c)/2$ quantile of the $v^*$, with linear interpolation, as `numpy.quantile`. NaN values are left out, e.g. an MCC that is undefined in a resample, or the difference of two infinite values in a comparison. The interval then holds only for the resamples where the value is defined. Their number is reported (`BootstrapInterval.num_undefined`, `ci_num_undefined` in `summary_long.csv`, `num_undefined` in the JSON documents, the `undefined` column of `irap-eval compare`), and `irap-eval` prints the intervals that leave out some, but not all, resamples. Infinite values are kept, e.g. the NLL of a resample in which a label has probability 0. A bound interpolated between an infinite value and a finite or equal value is that infinite value.

Intervals are computed for every metric of a result. Per-class metrics have no intervals. The same seed gives the same draws for every method scored on the same set.

## Comparing methods

`compare_methods(runs_a, runs_b)` (`irap-eval compare --a A --b B`, on the reference set, default $B = 1000$) recomputes the runs of both methods on the same resamples, which pairs the comparison over road sequences:

- The point difference is $\Delta = \bar m_a - \bar m_b$.
- Its interval is the percentile interval of the resampled differences $v^*_a - v^*_b$. The run terms of $a$ and $b$ use independent t draws.

There are no p-values. A difference is significant at level $1 - c$ if its interval excludes 0.

All runs of both methods must have the same segments, sequences, attributes and null policy.

## Interpreting results

- **Base conclusions on differences between methods on the same set.** Macro metrics (`mF1`, `_suppN`) depend on which classes the evaluation set contains, so a method's interval is not an interval for a population value.
- **`amF1` is the main metric.** Differences per attribute or for many metrics are exploratory, since some will exclude 0 by chance.
- **The intervals generalize to roads like those of the evaluated region, and to further runs of the methods.** The splits are by region, so they say nothing about other regions. For one region of a set, use `select_evaluation_subset`.
- **Invalid cells.** With `first_class`, every run is scored on the same segments, but an invalid cell counts as class 0, which is often, but not always, the most common class. With `exclude`, the scores measure skill when the response is usable, and must be read together with the number of invalid cells (`num_invalid` in the JSON documents).

Simulations on the Vietnam labels with synthetic predictors (2026-10-01, about 200 repetitions each) support the design of the intervals:

- **Run variation is not negligible.** With 3–5 runs per method, intervals of $\Delta$ amF1 over road sequences alone contained the true difference in only 18–55 % of the repetitions. With the t term, they did so in 97–100 %, so the intervals are conservative. A bootstrap over the runs, instead of the t term, reached only 80–89 %, since 3–5 values say little about their spread.
- **Single runs.** Intervals of the difference of two single runs covered the true difference in 85–96 %.
- **Single methods.** 95 % intervals of a single method's amF1 contained the value on all roads in 69–90 % of the repetitions, and those of `amF1_supp10` in 0–68 %.
- **Rejected alternatives.** Dirichlet weights (the Bayesian bootstrap) and bias-corrected intervals were not better, and in some cases worse.
