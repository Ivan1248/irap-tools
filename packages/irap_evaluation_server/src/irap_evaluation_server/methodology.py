"""The text of the Methodology page: how the server scores files and how to read the score
tables, with the values of the scoring settings. `irap_evaluation/docs/evaluation.md` has the
precise definitions.

The text is Markdown with TeX formulas between `$` (inline) or `$$` (display), which MathJax
typesets after the Markdown is converted (`MARKDOWN_EXTRAS`). The Markdown converter removes the
backslash before punctuation, so the formulas avoid `\\{`, `\\(` and `\\\\`, and `*`, which can
start emphasis.
"""

import dataclasses as dc
import math
import typing as T

import irap_evaluation as ie
from irap_data import get_dataset_preset
from irap_data.attrs import get_attrs_to_include

from .method_ranking import (
    DEFAULT_SORT_METRIC,
    NOT_WORSE_SIGNIFICANCE_LEVEL,
    PER_ATTRIBUTE_METRIC_NAMES,
)
from .scoring import ScoringSettings

#: The extras of the Markdown converter (markdown2) that the text needs: tables, and
#: 'code-friendly', which keeps the underscores of the formulas, e.g. `m_i`, from becoming emphasis.
MARKDOWN_EXTRAS = ["tables", "code-friendly"]

#: The ids of the sections that the notes of other pages link to (`routes.get_methodology_path`).
EVALUATION_SETS_SECTION = "evaluation-sets"
INTERVALS_SECTION = "intervals"
COMPARISONS_SECTION = "comparisons"


@dc.dataclass(frozen=True)
class MethodologySection:
    """A section of the Methodology page.

    Attributes:
        id: The id of its heading, the fragment of links to it.
        markdown: Its text, without the title.
    """

    id: str
    title: str
    markdown: str


def _format_percent(fraction: float) -> str:
    """Formats a fraction as a percentage without trailing zeros, e.g. '2.5%' or '95%'."""
    return f"{fraction * 100:g}%"


def _format_offset_window(offsets: T.Sequence[int]) -> str:
    """Formats context offsets as a range, e.g. '−4 to 0', or as a list if they have gaps."""
    if list(offsets) == list(range(min(offsets), max(offsets) + 1)):
        text = f"{min(offsets)} to {max(offsets)}"
    else:
        text = ", ".join(map(str, offsets))
    return text.replace("-", "−")


def _get_interval_probabilities(confidence: float) -> tuple[float, float]:
    """Returns the probabilities of the quantiles that bound a central interval with this
    confidence, e.g. (0.025, 0.975)."""
    upper_probability = (1 + confidence) / 2
    return 1 - upper_probability, upper_probability


def compute_t_quantile(probability: float, degrees_of_freedom: T.Literal[1, 2]) -> float:
    """Computes a quantile of Student's t-distribution with 1 or 2 degrees of freedom, which have
    closed forms."""
    if degrees_of_freedom == 1:
        return math.tan(math.pi * (probability - 0.5))
    if degrees_of_freedom == 2:
        return (2 * probability - 1) / math.sqrt(2 * probability * (1 - probability))
    raise ValueError(f"Only 1 or 2 degrees of freedom are supported, got {degrees_of_freedom}.")


_METHODS_MARKDOWN = r"""
- A **model** is one trained model instance. It has at most one prediction file per split.
- A **method** is a training configuration. Its models differ only in the seed.
- A **metric value** (score) is the value of a metric for a model on an evaluation set, computed
  from the model's predictions. The tables show metric values and their confidence intervals.
- The metric value of a method is the **mean over its models**: if a method has $M$ models, whose
  values of a metric on an evaluation set are $m_1, \dots, m_M$, its value is
  $\overline{m} = \frac{1}{M} \sum_{i=1}^{M} m_i$. On the model-compatible sets, models with
  different context offsets are not averaged, since their sets can differ.
"""


def _make_evaluation_sets_markdown(dataset_names: T.Sequence[str]) -> str:
    windows = "; ".join(
        f"{n}: {_format_offset_window(get_dataset_preset(n).reference_context_offsets)}"
        for n in dataset_names)
    return f"""
A model predicts the attributes of a segment from the images of the segment and of other segments
of the same road. Its **context offsets** are the positions of these segments in the road relative
to the predicted segment, e.g. 0 for the segment itself and −1 for the previous segment. An
**evaluation set** is a subset of the labeled segments of a split. The predictions of a file are
scored on one or two evaluation sets:

- **Reference set**: the segments for which every offset of a fixed window ({windows}) points to
  a segment of the same road with an image. It is the same for all models, so their metric values
  are comparable.
- **Model-compatible set**: the segments for which every offset of the model points to a segment of
  the same road with an image, as in the evaluation during training. It is shown only if it
  differs from the reference set. Models with different context offsets are not tested against
  each other on their model-compatible sets, since these can differ.

A model with an offset outside the reference window cannot predict all segments of the reference
set, so it is scored only on its model-compatible set. Files of unlabeled splits are not scored.
"""


def _make_attributes_markdown() -> str:
    return rf"""
- The scored attributes are the {len(get_attrs_to_include())} canonical iRAP attributes, without
  those that have no labels in the evaluation set. A segment without a label of an attribute is
  not scored for that attribute.
- The attribute filter selects the attributes that the attribute averages are computed over. It
  does not change the metric values of the individual attributes.
- An **attribute prediction** is the prediction of one attribute for one segment. It is
  **invalid** if the file leaves it empty, e.g. since a VLM's answer names no class of the
  attribute. In the metrics computed from predicted classes, an invalid attribute prediction counts
  as a prediction of the attribute's first class (in the dataset's class order). In NLL and Brier,
  it counts as the uniform distribution, which gives $\ln C$ and $1 - 1/C$ for an attribute with
  $C$ classes. Both are as in the evaluation during training. The Model and Method pages show the
  number of invalid predictions of each attribute.
- If a file does not predict an attribute, its attribute predictions count as invalid in all
  segments ("not predicted"), so that all models are scored on the same attributes, and leaving
  out an attribute does not remove it from the attribute averages.
"""


def _make_metrics_markdown() -> str:
    descriptions = "\n".join(f"- {ie.describe_metric(n)}" for n in PER_ATTRIBUTE_METRIC_NAMES)
    return rf"""
A metric of an attribute is computed from all segments of the evaluation set that have a label of
the attribute, pooled, e.g. from the attribute's confusion matrix over these segments.

A **macro** metric (`m…`) is a mean over classes, e.g.
$\mathrm{{mF1}} = \frac{{1}}{{|K|}} \sum_{{k \in K}} \mathrm{{F1}}_k$, where $\mathrm{{F1}}_k$ is
the F1 of class $k$ and $K$ is the set of the attribute's classes that occur as a label or as a
prediction in the evaluation set. (mNLL averages over the classes that occur as a label.) So a
macro metric depends on which classes the evaluation set contains, and rare classes make it noisy.
The suffix `_suppN`, e.g. `_supp10`, restricts $K$ to the classes with at least $N$ labeled
segments in the evaluation set.

An **attribute average**, with the prefix `a`, is the mean of a metric over the attributes, each
with the same weight, e.g. $\mathrm{{amF1}} = \frac{{1}}{{|A|}} \sum_{{a \in A}} \mathrm{{mF1}}_a$,
where $\mathrm{{mF1}}_a$ is the mF1 of attribute $a$ and $A$ is the set of the selected attributes
for which the metric is defined. The main metric is `{DEFAULT_SORT_METRIC}`. NLL, Brier and mNLL
need probabilistic predictions.

{descriptions}
"""


def _make_intervals_markdown(settings: ScoringSettings) -> str:
    confidence = _format_percent(settings.confidence)
    lower_probability, upper_probability = _get_interval_probabilities(settings.confidence)
    lower, upper = f"{lower_probability:g}", f"{upper_probability:g}"
    # The half-width of the interval of the term s / sqrt(M) · t relative to s.
    two_seed_width, three_seed_width = (compute_t_quantile(upper_probability, m - 1) / math.sqrt(m)
                                        for m in (2, 3))
    return rf"""
Below each metric value is its {confidence} confidence interval. It treats the roads of the
evaluation set as a random sample from a population of roads like them (of the same region), and
the models of a method as a random sample of the models that its training configuration gives with
different seeds. It is meant to contain the following **target** with probability {confidence},
i.e. {confidence} of the intervals computed in this way from such samples would contain it:

- for a method, the expected metric value over all seeds on the population of roads, not only the
  mean over its $M$ models,
- for a single model, its metric value on the population of roads.

The interval is computed from $B = {settings.num_resamples}$ resamples, which cover two sources of
variation:

1. **Roads.** The segments of a road are similar, so the roads are resampled, not the segments.
   Resample $b = 1, \dots, B$ draws $G$ roads with replacement from the $G$ roads of the evaluation
   set: road $g$ is drawn $w^{{(b)}}_g$ times, with
   $w^{{(b)}} \sim \mathrm{{Multinomial}}(G;\, 1/G, \dots, 1/G)$. The metric of the $i$-th model is
   computed again from the segments of the drawn roads, each road counted as often as it is drawn,
   which gives $m^{{(b)}}_i$. All models on an evaluation set use the same resamples.
2. **Seeds.** For a method with $M$ models, the resampled metric value of the method in resample
   $b$ is
   $$v^{{(b)}} = \overline{{m}}^{{(b)}} + \frac{{s^{{(b)}}}}{{\sqrt{{M}}}}\, t^{{(b)}},$$
   where $\overline{{m}}^{{(b)}}$ and $s^{{(b)}}$ are the mean and the sample standard deviation
   of $m^{{(b)}}_1, \dots, m^{{(b)}}_M$, and $t^{{(b)}}$ is drawn from Student's t-distribution
   $t_{{M-1}}$ with $M - 1$ degrees of freedom, independently for each resample and method (and the
   same for all metrics of the method). For a fixed set of roads, this gives the usual t-interval
   for the mean of $M$ normally distributed values. With a single model, $v^{{(b)}} = m^{{(b)}}_1$.

The interval runs from the ${lower}$ to the ${upper}$ quantile of $v^{{(1)}}, \dots, v^{{(B)}}$,
with linear interpolation. A `*` after it means that some resamples are left out, since the metric
is undefined in them, e.g. the MCC when all drawn labels of an attribute are of one class. Hover
over it for their number.

**The number of seeds.** With one model, the interval covers only the variation from the roads: it
cannot show how much models with other seeds would differ, so it understates the uncertainty about
the method's target. With $M$ models, the term $\frac{{s}}{{\sqrt{{M}}}}\, t$ alone gives an interval
with the half-width $t_{{M-1}}({upper})\, s / \sqrt{{M}}$, where $t_{{M-1}}({upper})$ is the ${upper}$
quantile of $t_{{M-1}}$ and $s$ is the standard deviation of the models' metric values. It is
${two_seed_width:.1f}\, s$ with two models, since $t_1$ has very heavy tails, and
${three_seed_width:.1f}\, s$ with three. From two models on, its expected value decreases with
each further model.

**Reliability.** The intervals are approximate. In simulations with synthetic predictions on the
Vietnam labels, in which the population was all roads of the training split and each evaluation
set a sample of 72 of them, 95% intervals contained their target in these fractions of the
repetitions:

| Interval of | Contained the target in |
|---|---|
| the difference between two methods with 3–5 models each (`amF1`) | 97–100% |
| the difference between two models | 85–96% |
| the `amF1` of one method | 69–90% |
| the `amF1_supp10` of one method | 0–68% |

The intervals of one method's metric value are often too narrow, since macro metrics depend on
which classes the evaluation set contains. Base conclusions on differences between methods (bold
and Δ), not on whether the intervals of two methods overlap.
"""


def _make_comparisons_markdown(settings: ScoringSettings) -> str:
    level = _format_percent(NOT_WORSE_SIGNIFICANCE_LEVEL)
    lower, upper = (f"{p:g}" for p in _get_interval_probabilities(settings.confidence))
    return rf"""
The metric values of methods are compared with each other in a column of the method table, or in a
row of the per-attribute table. Below, $\overline{{m}}$ is the metric value of a method,
$v^{{(b)}}$ its resampled metric value ([confidence intervals](#{INTERVALS_SECTION})), and
$\sigma = 1$ if a higher value of the metric is better, $\sigma = -1$ if a lower one is.

- **Shading**: the background of a metric value is darker the better the value is among the
  compared ones, in proportion to $(\overline{{m}} - \overline{{m}}_{{\mathrm{{worst}}}}) /
  (\overline{{m}}_{{\mathrm{{best}}}} - \overline{{m}}_{{\mathrm{{worst}}}})$. It is not a test.
- **Bold**: the best metric value, and those that are not significantly worse than it. With the
  resampled advantages $d^{{(b)}} = \sigma\, (v^{{(b)}} - v^{{(b)}}_{{\mathrm{{best}}}})$ of a
  method over the best one, the metric value is bold if
  $q_{{{1 - NOT_WORSE_SIGNIFICANCE_LEVEL:g}}}(d) \ge 0$, where $q_p$ is the $p$ quantile of
  $d^{{(1)}}, \dots, d^{{(B)}}$. This is a one-sided bootstrap test, at the {level} level, of
  whether the method is worse than the best one. It is **paired**: both methods are scored on the
  same resampled roads, so most of the variation between roads cancels in $d^{{(b)}}$, while the
  variation between the seeds of both methods remains. Values are bold only if all compared methods
  are scored on the same evaluation set: the reference set, or model-compatible sets with the same
  context offsets. The Method and Model pages shade their metric values without testing them.
- **Δ** ("Compare with", on the reference set): for a method and a reference method with the
  metric value $\overline{{m}}_{{\mathrm{{ref}}}}$, the difference
  $\Delta = \overline{{m}} - \overline{{m}}_{{\mathrm{{ref}}}}$. Its
  {_format_percent(settings.confidence)} interval runs from the ${lower}$ to the ${upper}$ quantile
  of the paired differences
  $v^{{(b)}} - v^{{(b)}}_{{\mathrm{{ref}}}}$, $b = 1, \dots, B$. "better" or "worse" means that the
  interval excludes 0, on the side where $\sigma \Delta > 0$ or $\sigma \Delta < 0$.

With many metrics, attributes or methods, some differences are significant by chance. Take
`{DEFAULT_SORT_METRIC}` as the main metric, and the others as exploratory.
"""


_SPLIT_USE_MARKDOWN = """
Metric values on a split that a model used are not of held-out data. The tables mark the methods
whose models used the shown split:

- **trained on the split**: the models were fitted to its labels. The Scores page hides these
  methods unless "Show methods trained on …" is checked.
- **early stopping on the split**: the split was used to choose when to stop training or which
  checkpoint to keep, so the scores are optimistic.
- **training splits unknown**: the models may have been trained on the split.
"""


def make_methodology_sections(settings: ScoringSettings,
                              dataset_names: T.Sequence[str]) -> list[MethodologySection]:
    """Makes the sections of the Methodology page.

    Args:
        dataset_names: The configured datasets, whose reference windows the text gives.

    Raises:
        ValueError: If the settings score invalid attribute predictions or missing attributes
            otherwise than the text says.
    """
    if (settings.null_policy, settings.missing_attribute_policy) != ("first_class", "invalid"):
        raise ValueError(f"The methodology text describes the null policy 'first_class' and the"
                         f" missing attribute policy 'invalid', not {settings.null_policy!r} and"
                         f" {settings.missing_attribute_policy!r}.")
    return [
        MethodologySection("methods", "Models and methods", _METHODS_MARKDOWN),
        MethodologySection(EVALUATION_SETS_SECTION, "Evaluation sets",
                           _make_evaluation_sets_markdown(dataset_names)),
        MethodologySection("attributes", "Attributes and invalid predictions",
                           _make_attributes_markdown()),
        MethodologySection("metrics", "Metrics", _make_metrics_markdown()),
        MethodologySection(INTERVALS_SECTION, "Confidence intervals",
                           _make_intervals_markdown(settings)),
        MethodologySection(COMPARISONS_SECTION, "Comparing methods",
                           _make_comparisons_markdown(settings)),
        MethodologySection("split-use", "Splits that a model used", _SPLIT_USE_MARKDOWN),
    ]
