# Study guide: not-strangers, ground up

This document exists so you can walk into a technical interview and explain
every number, every design decision, and every mistake this project found in
itself — not because you memorized a script, but because you actually
understand the mechanics underneath. It assumes you know that AI stands for
artificial intelligence and ML for machine learning, and nothing more than
that. Every other term — gradient boosting, precision, recall, a graph, a
Shapley value, temporal leakage, class imbalance — is defined in plain
language the first time it's used, before you need it.

Every number in this document is quoted from a file in `results/`. Every
code excerpt is quoted from a file that actually exists in this repository —
`src/*.py`, `tests/*.py`, `app.py`, `dashboard_attribution.py`. Nothing here
describes code that wasn't opened and read.

---

# PART 1 — FOUNDATIONS

## 1.1 What a machine learning model actually is

Strip away the mystique: a machine learning model is a function. It takes
some numbers in, and produces a number out. That's it. The "learning" part
is about *how the function got its shape*, not about anything the function
does at the moment you use it.

Let's make this concrete with a real row from this project's dataset. Every
row in `train_transaction.csv` is one payment transaction. Transaction
`2987000` looks like this (queried directly from the real file):

```
TransactionID     2987000
isFraud                 0
TransactionDT        86400
TransactionAmt        68.5
ProductCD                W
card1                13926
card2                  NaN
card3                150.0
card4             discover
card5                142.0
card6               credit
addr1                315.0
addr2                 87.0
P_emaildomain          NaN
C1                     1.0
C2                     1.0
D1                    14.0
D2                     NaN
```

`isFraud` is the **label** — the true answer, known only because this
transaction eventually resulted (or didn't) in a chargeback. Every other
field is a **feature** — a piece of information about the transaction that
existed *before* anyone knew the answer. `TransactionAmt` is the amount in
dollars. `ProductCD` is a coded product category. `card1`-`card6` are
numeric/categorical codes describing the payment card (issuing bank, card
network, card type — the raw values are anonymized, so `card4=discover`
tells you the network but `card1=13926` is just an opaque ID). `addr1`/`addr2`
are coded address regions. `C1`, `C2` are anonymized counting features
(count of something related to this card — the competition doesn't say
exactly what, on purpose, to prevent identity re-engineering). `D1`, `D2` are
anonymized *time-delta* features — "days since X happened" for some X the
competition also won't name (this project's own investigation figured out
what `D1` almost certainly measures — see Part 3).

A **model**, mechanically, is:

1. A fixed set of **input features** it expects (here: hundreds of columns
   like the ones above).
2. A large number of **learned parameters** — internal numbers, tuned
   during training, that are *not* part of the input. You never see them
   directly; they encode "how much each feature matters, and in what
   combination."
3. A **prediction function** that combines the inputs and the parameters
   using fixed arithmetic (addition, multiplication, comparisons,
   thresholds — for the model type used in this project, a large number of
   simple yes/no branching rules, explained in 1.2) to produce one number
   out.

For this project's model, the output is one number between 0 and 1 — an
**abuse score**, higher meaning "the model thinks this transaction looks
more like fraud." Feed it transaction `2987000`'s features, and it outputs
some number, say 0.003. Feed it a different transaction's features — a real
fraud example from this same file, `TransactionID 2987203`
(`TransactionAmt=445.0`, `ProductCD=W`, `card4=visa`, `card6=credit`,
`P_emaildomain=aol.com`, `C1=2.0`, `D1=57.0`) — and it might output 0.62.
The model doesn't "know" what fraud is in any human sense. It has learned,
from thousands of past examples where the true answer was known, which
*combinations* of feature values tended to co-occur with `isFraud=1`, and it
reproduces that pattern on new rows it's never seen.

The actual code that does this, from `src/model.py`:

```python
def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
```

That's the entire "prediction" step, mechanically — one line, calling into
the trained model object (`lgb.Booster` — explained in 1.2). Everything that
makes the number meaningful happened earlier, during **training**.

## 1.2 What "training" means mechanically, and what LightGBM does differently from a single decision tree

**Training** is the process of choosing the model's internal parameters by
showing it many examples where the answer is already known, and adjusting
the parameters to make the model's outputs match those known answers more
closely.

The simplest possible model of this kind is a **decision tree**: a sequence
of yes/no questions about the input, arranged in a branching structure, that
ends in a prediction. For example, a (deliberately tiny, illustrative — not
the real model) tree might be:

```
Is TransactionAmt > 500?
├── Yes: Is ProductCD == "C"?
│         ├── Yes: predict 0.40
│         └── No:  predict 0.05
└── No:  Is cluster_prior_fraud_share > 0.3?
          ├── Yes: predict 0.55
          └── No:  predict 0.01
```

"Training" a decision tree means the algorithm looks at every possible
question it could ask at each branch point (every feature, every possible
split value) and picks whichever single question best separates the fraud
rows from the non-fraud rows at that point, then repeats the process inside
each resulting branch. This is a *greedy, local* procedure — at each step it
takes the single best split available right now, without looking ahead.

A single decision tree has a real weakness: it can only slice the data along
axis-aligned rectangular regions, and given enough branches, it will
eventually carve out a tiny region containing just one or two training
examples and predict them perfectly — which tells you nothing useful about
new data (see 1.6, overfitting).

**Gradient boosting** is a different strategy: instead of building one
deep, complicated tree, build many *very small, weak* trees, one after
another, where each new tree's entire job is to correct the mistakes the
trees built so far are still making. Concretely: build tree 1, see where it
over- or under-predicts on the training data, then build tree 2 whose job is
specifically to predict *that error* (not the original label), and add
tree 2's prediction (scaled down by a small factor, the **learning rate**)
onto tree 1's. Repeat this hundreds of times. The final prediction is the
sum of every small tree's contribution. Each individual tree is weak and
unimpressive; the sum of hundreds of them, each fixing the previous
ensemble's specific residual mistakes, is powerful.

**LightGBM** is a specific, fast implementation of gradient boosting over
decision trees, and it's the library this project uses (`import lightgbm as
lgb` in `src/model.py`). Here is the actual training code:

```python
SEED = 42
NUM_BOOST_ROUND = 300

LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "None",
    "verbosity": -1,
    "seed": SEED,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
}

def _fit(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    categorical = X_train.select_dtypes(include="category").columns.tolist()
    dataset = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical or "auto"
    )
    return lgb.train(LGBM_PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)
```

Reading this line by line: `X_train` is the feature matrix — one row per
transaction, one column per feature (`TransactionAmt`, `card1`, and so on).
`y_train` is the matching column of true labels (0 or 1). `objective:
"binary"` tells LightGBM this is a yes/no prediction problem (fraud or not),
so internally it will optimize using **log loss** (a way of scoring
predictions that punishes confident-and-wrong predictions much more
severely than a plain miss — full mathematical detail isn't needed for this
project, but the intuition matters: it pushes the model toward well-ranked,
appropriately-confident scores). `num_leaves: 63` caps how complex each
individual small tree is allowed to get (more leaves = more splits = a more
complex tree). `learning_rate: 0.05` is how much each new tree's correction
gets scaled down before being added to the running total — small steps,
many of them, tend to generalize better than few large steps.
`num_boost_round: 300` means 300 small trees get built, one after another.
`feature_fraction: 0.8` and `bagging_fraction: 0.8` mean each tree only gets
to look at a random 80% of the columns and 80% of the rows, respectively —
this is a defense against any one feature or any one cluster of rows
dominating every single tree, which would make the whole ensemble too
reliant on one signal. `min_data_in_leaf: 50` says a branch can't end in a
leaf covering fewer than 50 training rows — without this, the tree could
carve out a leaf covering exactly 2 rows and memorize them, which is
overfitting (1.6) in miniature. `seed: 42` fixes the random-number generator
so that every random choice LightGBM makes (which 80% of features, which
80% of rows) is reproducible — run this exact code twice, get the exact
same trained model both times.

This project trains **two** models this way — a "baseline" model and a
"cluster" model — using the identical `_fit()` function, identical
`LGBM_PARAMS`, identical seed, identical `num_boost_round`. The *only*
difference between them is which columns are in `X_train`. This is not
incidental; it's the entire point of the project's central experiment (the
ablation — Part 6). The model.py docstring says this explicitly:

> "train_baseline_model and train_cluster_model both call the same private
> `_fit()` helper with the same LGBM_PARAMS dict, the same NUM_BOOST_ROUND
> and the same seed — structurally, not just by value, so the only way the
> two models can differ is in the columns of X. That is the entire point of
> the ablation: if the two training functions diverged in any other way,
> the comparison would measure that divergence, not the cluster features."

## 1.3 What a prediction score between 0 and 1 means, and why it isn't a probability unless the model is calibrated

The model's raw output, before any conversion, lives in what's called
**log-odds** or **margin** space — a number that can be any real value
(negative, zero, or positive), where more positive means "more confident
this is fraud" and more negative means "more confident this is not." To
turn that into the familiar 0-to-1 number, you pass it through a **sigmoid
function**, `1 / (1 + e^(-x))`, which squashes any real number into the
(0, 1) range while preserving order (a bigger margin always produces a
bigger 0-to-1 score).

Here's the critical distinction, and it's one this project measured
directly rather than assuming: **a 0-to-1 score is not automatically a
probability.** A probability has a specific meaning — "if you gather 100
transactions that all scored 0.30, about 30 of them should actually turn
out to be fraud." A model's raw score only has that property if the model
is **calibrated**: trained and/or adjusted so that its output distribution
actually matches real-world outcome frequencies at every score level.
Nothing about ordinary training (1.2) guarantees this. A model can be
extremely good at *ranking* transactions from least to most suspicious
(which is what PR-AUC in Part 1.5 measures) while being badly wrong about
the actual probability its own numbers imply.

This project checked its own cluster model's calibration directly (see Part
9 for the full mechanics) and found it is **not** well calibrated in the
region that matters: the highest-score bin has a mean predicted score of
0.4795, but the real observed fraud rate among transactions in that bin is
only 0.3509 — the model is overconfident by about 0.13 exactly where its
decisions get made. This is why `src/policy.py`'s own docstring is careful
to describe its thresholds as "cost-minimizing," not "probability
estimates" — and why `results/ablation.md`'s Calibration section states
plainly: *"policy.py's REVIEW_THRESHOLD (0.1843) should be read as an
arbitrary cut on this model's score scale, not as 'we estimate >18.43%
abuse risk.'"*

## 1.4 Class imbalance: why 3.5% fraud makes accuracy useless

**Class imbalance** means the two outcomes you're trying to predict occur at
very different rates. Here's the real number for this dataset, computed
directly from the raw file:

- Total transactions: **590,540**
- Fraud transactions (`isFraud == 1`): **20,663**
- Fraud rate: **3.499%** — call it 3.5%.

Now consider **accuracy** — the most intuitive-seeming metric: "what
fraction of predictions did the model get right?" Imagine a deliberately
useless model that *always* predicts "not fraud," no matter what. What's
its accuracy?

It's right every single time on the 569,877 legitimate transactions, and
wrong every single time on the 20,663 fraud transactions. Accuracy =
569,877 / 590,540 = **96.5%**.

A model that never once catches a single fraud case — a model that is,
functionally, useless for the entire purpose of this project — scores
96.5% accuracy. That number *sounds* excellent and *is* worthless. This is
the whole problem with accuracy under class imbalance: when one class
dominates, a model can achieve a high accuracy score purely by ignoring the
rare class entirely, and the metric cannot tell the difference between "a
genuinely excellent fraud detector" and "a detector that never detects
anything." Accuracy is simply the wrong tool to reach for whenever the two
outcomes aren't roughly balanced, and 96.5%-vs-3.5% is about as imbalanced
as it gets.

## 1.5 Precision, recall, false positives, false negatives — then PR-AUC vs. ROC-AUC

Since accuracy fails, fraud detection is evaluated with a different set of
concepts, built around a 2x2 table of what actually happened vs. what the
model predicted, at some chosen score threshold:

|  | Model says "fraud" | Model says "not fraud" |
|---|---|---|
| **Actually fraud** | True Positive (TP) | False Negative (FN) |
| **Actually not fraud** | False Positive (FP) | True Negative (TN) |

A **false positive** is a legitimate transaction the model wrongly flags —
the cost is customer friction: an unnecessary step-up challenge, a held
transaction, an annoyed real customer. A **false negative** is real fraud
the model misses entirely — the cost is the actual financial loss from that
fraud going through undetected.

**Precision** answers: *of everything the model flagged, how much was
actually fraud?* Formula: `TP / (TP + FP)`. High precision means when the
model raises an alarm, it's usually right — few false alarms wasting
analyst time.

**Recall** answers: *of everything that actually was fraud, how much did
the model catch?* Formula: `TP / (TP + FN)`. High recall means the model
misses very little real fraud, even if it means raising more false alarms
along the way.

These two trade off against each other, and the trade-off is controlled by
where you set the score threshold. Set the threshold very low (flag almost
everything) and recall shoots toward 100% (you catch nearly all fraud) while
precision collapses (almost everything you flagged was actually fine). Set
the threshold very high (flag almost nothing) and precision goes up while
recall collapses.

A concrete worked example, using this project's own real cluster-model
numbers from `results/ablation.md`'s cost-curve section: at the
cost-minimizing threshold of 0.0103, the cluster model achieves **recall of
0.9530** and a **false positive rate of 0.3695** on the test set. Reading
that plainly: it catches 95.3% of real fraud, at the cost of also flagging
almost 37% of every legitimate transaction. That specific trade-off point
was chosen deliberately (see Part 9) because of an assumed 100:1 cost ratio
between missing fraud and annoying a legitimate customer — a different cost
assumption would move that threshold, and therefore this whole trade-off,
substantially.

**PR-AUC** ("precision-recall area under the curve") summarizes this
trade-off into one number without picking any single threshold: it plots
precision (y-axis) against recall (x-axis) as the threshold sweeps from
"flag everything" to "flag nothing," and reports the area under that curve.
A perfect classifier — one that ranks every fraud case above every
legitimate case — scores 1.0. A model that ranks completely randomly scores
approximately the base rate (here, ~0.035), *not* 0.5 — this project's own
test (`tests/test_evaluate.py`) checks exactly this:

```python
def test_pr_auc_uncorrelated_scores_are_near_base_rate():
    rng = np.random.default_rng(0)
    n = 20_000
    base_rate = 0.035
    y_true = (rng.random(n) < base_rate).astype(int)
    y_score = rng.random(n)  # uncorrelated with y_true

    score = pr_auc(y_true, y_score)

    assert score == pytest.approx(base_rate, abs=0.02)
```

This project's headline number is exactly this metric: the baseline model
scores **0.5646** PR-AUC, the cluster model scores **0.6322** — both far
above the ~0.035 random-guessing floor, and the cluster model measurably
ahead of the baseline (the entire subject of Part 6).

**ROC-AUC** is a related, more commonly-known metric that plots recall
(true positive rate) against the false-positive rate instead. The reason
this project uses PR-AUC as its headline metric instead of ROC-AUC is
specifically because of class imbalance: with 96.5% of transactions
legitimate, the "true negative" pool is enormous, and ROC-AUC's
false-positive-rate axis is measured *against that huge pool*
(`FP / (FP + TN)`), so it can look deceptively good even when precision (the
number that actually matters to an analyst drowning in false alarms) is
poor — a model can have a great-looking ROC-AUC while still flagging so
many false positives in absolute terms that it's operationally useless.
PR-AUC's precision axis is sensitive to exactly the failure mode that
matters here: how much of an analyst's actual workload, if they acted on
every flag, would be wasted on false alarms. That's why the entire ablation
in this project (Part 6) is reported in PR-AUC, not ROC-AUC.

## 1.6 Overfitting and leakage: the central danger in this project specifically

**Overfitting** happens when a model doesn't just learn the general pattern
in the training data — it also memorizes the specific noise and coincidence
in that particular sample. An overfit model looks excellent on the data it
was trained on and performs much worse on new data it hasn't seen, because
it learned quirks of the training sample that don't generalize. The
defense, universally, is to **evaluate on data the model never saw during
training** — a held-out test set.

**Leakage** is a more insidious and project-specific danger: it's when
information that would not actually be available at prediction time
sneaks into the features or the evaluation, making the model look far
better than it would in a real deployment. The clearest example: if you
accidentally include a feature computed *using the label itself*, or
computed using *future* information that wouldn't exist yet when a real
decision has to be made, your reported metrics measure an illusion — the
model isn't actually predicting anything, it's just reading a hint that
happens to already contain the answer.

This is **the central danger in this project specifically**, for two
structural reasons unique to how it's built:

1. **The graph/clustering step aggregates information across many
   transactions belonging to the same entity, and time is not naturally
   respected by an aggregation.** If a cluster's summary statistic (say,
   "has any member of this cluster ever committed fraud") is computed once
   over *all* of a cluster's transactions — including ones that happen
   later in time than the transaction currently being scored — that
   feature would be leaking the future into the present. A transaction from
   January could get a feature computed partly from May's data, which is
   information that would not exist yet if this system were actually
   running live in January.

2. **The data must be split by time, never randomly, and every downstream
   step must respect that split.** Randomly shuffling transactions before
   splitting into train/test would scatter each entity's transactions
   across both sets — meaning the model could train on some of an entity's
   later transactions and be tested on that same entity's earlier ones,
   which is not a real test of predicting the *future* from the *past*, the
   actual real-world task this system exists to do.

This is precisely why `CLAUDE.md` (the project's own binding rules) states,
verbatim: *"Splits are ALWAYS temporal on TransactionDT. Never random"* and
*"Cluster features must be computed causally. Graph structure and
aggregates for a test transaction may only use transactions with strictly
earlier TransactionDT."* Part 5 walks through exactly how this project
enforces both rules mechanically, and how it proved — not just asserted —
that its own dominant feature doesn't violate them.

---

# PART 2 — THE DATA

## IEEE-CIS structure: two files, joined, unevenly

This project uses the IEEE-CIS Fraud Detection dataset (a real, published
Kaggle competition dataset). It ships as two separate CSV files that must be
joined together:

- **`train_transaction.csv`**: 590,540 rows, 394 columns. One row per
  payment transaction. Contains the label (`isFraud`), the transaction
  amount and product code, the card/address/email fields, and hundreds of
  anonymized behavioral features (the `C`, `D`, `M`, and `V` columns below).
- **`train_identity.csv`**: 41 columns. One row *per transaction that has an
  identity record* — device and connection information (browser, device
  type, screen resolution class, and so on) collected at the time of that
  specific transaction.

Critically, **not every transaction has an identity record.** Verified
directly against the real files: `train_identity.csv` has 144,233 rows,
against 590,540 transactions — a coverage of **24.42%**, matching this
project's own stated "~24%" figure exactly. `src/data.py`'s
`load_transactions` function joins the two files with a **left join** (keep
every transaction row, attach identity data where it exists, leave identity
columns null where it doesn't) rather than an **inner join** (keep only rows
present in both), specifically because an inner join would silently discard
the other 75.58% of transactions:

```python
def load_transactions(path: Path, nrows: int | None = None) -> pd.DataFrame:
    ...
    merged = transactions.merge(identity, on="TransactionID", how="left")
```

The docstring states the reasoning directly: *"Uses a left join (not
inner): train_identity.csv covers only ~24% of TransactionIDs, so an inner
join would silently discard the other ~76% of rows."* This single design
choice has a real downstream consequence for entity resolution (Part 3):
since device information is missing for three-quarters of transactions, any
identity-resolution strategy that *required* a device field would fail on
most of the data — which is exactly why this project's primary linkage key
avoids the identity file entirely.

After joining, `_downcast_dtypes` shrinks the resulting DataFrame's memory
footprint: every `float64` column becomes `float32` (halving its memory
with negligible precision loss for this purpose), and every text
(`object`-typed) column with 50 or fewer distinct values becomes a
`category` (pandas' memory-efficient encoding for columns that repeat the
same small set of strings over and over, like `ProductCD` which only ever
takes values `W`, `C`, `R`, `H`, `S`). The real effect, printed by the
pipeline on every run: raw memory usage of about 1938 MB drops to about
1016 MB after downcasting — roughly halved, for a file that would otherwise
strain a typical laptop's memory just to load.

## What each column family means

- **`card1`-`card6`**: attributes of the payment card and card network.
  `card4` and `card6` are human-readable categories (`visa`/`discover`,
  `credit`/`debit`); `card1`, `card2`, `card3`, `card5` are anonymized
  numeric codes. Verified directly: `card3` and `card5` are dominated by a
  small number of values across the whole dataset — this project's own
  investigation (Part 3/4) found that among the 20 largest resolved
  identities, `card3` and `card5` never varied at all, confirming they
  behave as sub-attributes of the card (issuing bank/network category)
  rather than independent identifying information.
- **`addr1`, `addr2`**: coded address/region fields — `addr1` is a
  finer-grained region-level code (hundreds of distinct values across the
  dataset), `addr2` is coarser (closer to a country-level code, almost
  constant for this dataset's population).
- **`D1`-`D15`** (15 columns): anonymized *time-delta* features — each one
  is "number of days since some event," where the competition deliberately
  does not disclose what event each specific `D` column measures, to
  prevent the anonymization from being reverse-engineered. This project's
  own investigation (Part 3) worked out, from first principles and then
  confirmed against real data, that `D1` behaves exactly like "days since
  this card was first seen."
- **`C1`-`C14`** (14 columns): anonymized *counting* features — each is a
  count of something related to the card/address/email combination (again,
  the competition does not disclose exactly what is being counted).
- **`M1`-`M9`** (9 columns): anonymized *match* flags — each is
  effectively a yes/no/unknown flag for whether some pair of fields
  matched (for example, whether the billing and shipping address matched;
  the competition does not confirm the exact pairing for every column).
- **`V1`-`V339`** (339 columns — by far the largest column family):
  Vesta's own proprietary, fully anonymized engineered features, built by
  the company that contributed this dataset from richer underlying
  signals the competition does not expose at all. Their meaning is opaque
  by design, but they carry real predictive signal — this project's own
  feature-importance table (Part 6) shows several `V` columns, notably
  `V258`, among the highest-importance features in the trained model,
  second only to this project's own engineered `cluster_prior_fraud_share`.

## How the fraud labels were created, and why they propagate across a card

`CLAUDE.md` states the label mechanism this project must treat as ground
truth: *"Labels are chargeback-reported and propagate across a card once
reported — treat as noisy."* Mechanically: `isFraud=1` means a chargeback
was eventually reported against that transaction (or, per the competition's
own documented labeling process, against the *card* involved, at which
point every transaction that card is known to have made in a defined
window gets retroactively marked `isFraud=1` too — not because each of
those individual transactions was independently confirmed fraudulent, but
because the card itself was flagged and the label propagated backward
across its history).

This has a consequence that shapes this entire project's central, most
important finding (Part 6 and Part 7): a feature that measures "has this
card/identity already been flagged as fraud before" is not discovering
*new* abuse — it is close to directly re-reading the *label-generation
process itself*. That doesn't make such a feature invalid (it is not
leakage in the technical sense of Part 5 — it never reads a label from the
future relative to the transaction being scored) but it does mean its
predictive power says more about "does this system remember who was
already caught" than "can this system independently discover a
never-before-seen abuse pattern." This distinction — a legitimate,
non-leaking feature whose power is nonetheless partly circular — is this
project's single most important, most carefully argued finding, and it
recurs throughout Parts 6, 7, and 10.

---

# PART 3 — ENTITY RESOLUTION

## The problem: no customer ID exists

IEEE-CIS's data has no customer ID column. There is no field that says
"transaction #4,102,559 and transaction #6,881,204 were made by the same
person." This is deliberate, both for the competition's anonymization
requirements and because it mirrors a genuinely common real-world payments
problem: a payment processor sees individual transactions, not verified
customer accounts, and has to infer which transactions likely belong
together before it can reason about *coordinated, multi-transaction*
abuse rather than judging each transaction in total isolation.

`src/entities.py` solves this by constructing a **uid** (a persistent,
synthetic identity key) for each transaction, out of fields that already
exist in the transaction file:

```python
uid = card1_addr1_origin_day
```

where `origin_day` is derived from two other fields, explained next.

## What D1 measures, and why day - D1 is constant for a card

Recall from Part 2: `D1` is one of the competition's anonymized time-delta
columns, and its exact meaning is not disclosed. This project worked out
what it almost certainly measures by reasoning from its behavior, then
confirmed that reasoning against real data (documented in
`results/d1_investigation.md`).

The hypothesis: `D1` = "number of days since this card was first seen in
the data." If that's true, then for any given transaction:

```
day        = TransactionDT // 86400          (whole days since the dataset's own epoch)
origin_day = day - D1                        (the day this card was FIRST seen)
```

Since `86400` is the number of seconds in a day, integer-dividing
`TransactionDT` (seconds since some fixed reference point) by `86400`
converts a timestamp into a whole day-number. If `D1` truly counts "days
since first seen," then subtracting it from the current day-number should
land on the *same* value — the card's first-seen day — no matter which of
that card's transactions you compute it from, because as the calendar day
advances, `D1` advances by exactly the same amount.

Here is the actual arithmetic, using three real rows pulled directly from
`train_transaction.csv`, all three of which share `card1=15775, addr1=330`
(this is uid `15775_330_129`, the single largest resolved identity in the
whole dataset — 1,414 transactions):

| TransactionID | TransactionDT | day (DT // 86400) | D1 | origin_day (day − D1) |
|---|---:|---:|---:|---:|
| 3428022 | 11,204,367 | 129 | 0.0 | **129** |
| 3455679 | 12,081,968 | 139 | 10.0 | **129** |
| 3465818 | 12,426,792 | 143 | 14.0 | **129** |

Three different transactions, on three different calendar days (129, 139,
143), with three different `D1` values (0, 10, 14) — and every single one
resolves to the exact same `origin_day`, 129. That's the whole mechanism:
as real calendar time advances by 10 days (day 129 → 139), this card's `D1`
also advances by exactly 10 days (0 → 10), because both are counting from
the same fixed starting point — the day this card first appears. Subtracting
one from the other cancels out the passage of time entirely and leaves a
constant: the card's origin day. That constant is what makes `origin_day`
usable as part of a persistent identity key — no matter which of the card's
1,414 transactions you look at, spread across the whole dataset, this
arithmetic always lands on 129.

Combined with `card1` and `addr1` (the address the card was used with),
this produces `uid = "15775_330_129"` — a synthetic identity that groups
every transaction from this card/address/origin combination into one
persistent entity, from `src/entities.py`'s actual implementation:

```python
def resolve_entities(transactions: pd.DataFrame) -> pd.Series:
    keys = extract_entity_keys(transactions)

    valid = (
        keys["card1"].notna() & keys["addr1"].notna() & keys["origin_day"].notna()
    )

    def _int_str(col: pd.Series) -> pd.Series:
        return col.round().astype("Int64").astype(str)

    uid = (
        _int_str(keys["card1"])
        + "_"
        + _int_str(keys["addr1"])
        + "_"
        + _int_str(keys["origin_day"])
    ).where(valid)
    uid.name = "uid"
    return uid
```

Note the null handling: if `card1`, `addr1`, or `D1` is missing for a given
transaction, that transaction gets `NaN` as its uid — never a fallback
guess, never a dropped row (the module docstring states this explicitly:
*"if card1, addr1 or D1 is null for a transaction, that transaction gets no
uid (NaN) rather than a fallback value or a silently dropped row"*). Why
this matters is covered later in this Part.

## Why 30% of rows get a negative origin_day, and what that revealed

A `day - D1` subtraction can go negative, and it does — often. Verified
directly against the real data: of 590,540 rows, **179,300 (30.36%)** have
a negative `origin_day`, **408,917 (69.24%)** have a positive one, **1,054
(0.18%)** are exactly zero, and **1,269 (0.21%)** are null (`results/d1_investigation.md`).

A negative `origin_day` means: the card's `D1` (days since first seen) is
*larger* than how far back the dataset's own recorded window goes. Here's a
real, exact example, again pulled directly from the raw file — uid
`12695_325_-342`:

| TransactionID | TransactionDT | day (DT // 86400) | D1 | origin_day (day − D1) |
|---|---:|---:|---:|---:|
| 2990208 | 153,379 | 1 | 343.0 | **−342** |

This is one of the very first transactions in the entire dataset — day 1 of
an observed window that (verified directly) spans from day 1 to day 182,
roughly six months. And on that very first day, this card's `D1` already
says it's 343 days old — nearly a full year. The card didn't spring into
existence on day 1; it was already old when the observation window began.
`origin_day` simply reports how far *before* day 0 the card must have
originated: `1 - 343 = -342`, i.e. this card's true first-seen day was 342
days before this dataset started recording anything. This is not a bug in
the derivation — it's exactly what you'd expect from a dataset that only
observes a fixed ~6-month slice of an ongoing world where cards existed
before that window opened. This particular uid, incidentally, is also one
of the ten largest *label-impure* identities in the dataset (123
transactions, 73.17% fraud rate — see the collision check below).

This project checked whether negative-`origin_day` rows are meaningfully
different in kind, not just in sign, and found real differences (all from
`results/d1_investigation.md`):

- Fraud rate: negative-group **1.60%**, positive-group **4.33%**.
- Identity-record coverage: negative-group **7.35%**, positive-group
  **31.97%** — negative-origin_day cards are far less likely to have an
  identity record at all.
- Label purity of multi-transaction uids: negative-group **99.24%**
  weighted, positive-group **96.61%** weighted — negative-origin_day
  identities resolve into *more* internally-consistent clusters than
  positive-origin_day ones.

So the sign of `origin_day` isn't cosmetic — it correlates with genuinely
different transaction behavior and with how trustworthy the resulting uid
is as a label-consistent grouping.

## Label purity: what it tests, why 97.61% is the number that matters

**Label purity**, for a resolved identity (uid), asks: *do all the
transactions grouped under this one uid actually share the same fraud
label?* If a uid genuinely represents one consistent entity engaging in
one consistent kind of behavior, you'd expect its transactions to be
label-consistent — either the entity is fraudulent and (mostly) all its
transactions get caught, or it isn't and none do. A uid whose transactions
are a scattered mix of fraud and non-fraud is either (a) genuinely a mixed
bag by coincidence, or (b) merging multiple distinct real-world entities
under one synthetic key — the over-merging risk this Part's final section
addresses.

Measured against the real data, over the 83,557 uids with 2 or more
transactions (`results/uid_validation.md`):

- **Unweighted label-pure fraction: 98.53%** — the share of *uids* (each
  counted once, regardless of size) that are internally label-consistent.
- **Weighted label-pure fraction: 97.61%** — the share of *transactions*
  (not uids) that fall inside a label-pure cluster.

The distinction matters and is worth internalizing: unweighted treats a
1,414-transaction uid the same as a 2-transaction one — one vote each.
Weighted asks, "of all the transaction volume sitting inside multi-member
clusters, what fraction of that volume is inside a clean cluster?" These
two numbers are close together here (98.53% vs. 97.61%), which itself is
informative: if they'd differed by a lot, it would mean purity depends
heavily on cluster size (e.g., only small clusters are pure, large ones are
messy) — they don't, so purity holds up reasonably evenly across cluster
sizes.

**What a low number would have meant:** if weighted label purity had come
in low — say, 60% — it would mean the `card1_addr1_origin_day` key is
mostly noise: a large fraction of the time, it's lumping together
transactions that don't actually share an outcome, which would make every
downstream cluster-level feature (Part 4) close to random with respect to
fraud, and this project's entire premise (that entity resolution followed
by cluster-level scoring adds real signal) would have collapsed at the
very first step. 97.61% instead means the key, whatever its other flaws,
overwhelmingly groups transactions that *do* share an outcome — a necessary
(though not sufficient — see below) condition for the rest of the pipeline
to make sense.

## The over-merging finding, and why it was kept rather than fixed

High label purity answers "does a uid's transactions agree with each other"
— it does *not* answer "does a uid actually correspond to one real
person." Those are different claims, and this project checked the second
one directly with a **collision check**: for the 20 largest resolved
identities, count how many *distinct* values of `card2`, `card3`, `card5`,
and `P_emaildomain` (email domain) appear within each uid's own rows. If a
uid is genuinely one client, these secondary fields should be near-constant
(a real person keeps the same card sub-attributes and, mostly, the same
email address).

`card2`, `card3`, and `card5` were indeed constant (exactly one distinct
value) across all 20 largest uids — but that's expected and not very
informative, since (per Part 2) `card3`/`card5` are essentially fixed
sub-attributes of the card itself, not independent per-person signals.

`P_emaildomain` is the meaningful test, and the result is stated plainly in
`results/d1_investigation.md`: **10 of the 20 largest uids show more than
one distinct email domain.** That means `card1_addr1_origin_day` is not
unique to a single cardholder for at least half of the dataset's very
largest identities — distinct real people, using different email addresses,
are being merged into one synthetic uid because they happen to share a
card1, an address, and a first-seen day.

The project's own framing of this finding is precise and worth quoting
directly, because it's the kind of nuance an interviewer will specifically
probe: *"Label purity being high... shows these merges are usually
*label-consistent* (the merged clients mostly share the same isFraud
outcome), which is a much weaker claim than the uid being *correct* — i.e.
actually identifying one physical client. This is stability without
correctness: safe enough to use as a clustering key for label-consistent
aggregation, but it should not be presented as 'the client' in an
investigator-facing explanation without that caveat."*

Why keep an identity key that's known to over-merge, rather than fix it?
Because in this project's specific context — detecting *coordinated* abuse
— over-merging multiple people who share a card, address, and first-seen
day is not obviously a defect. If several distinct email addresses are
transacting through the same card/address combination, that is itself a
pattern worth surfacing to an investigator (it could be a shared household,
or it could be exactly the kind of coordinated card-testing/mule activity
this project exists to catch) — collapsing it into one investigable unit is
arguably the right behavior, not a bug to engineer away. `README.md`
states this explicitly as a stated limitation, not a hidden flaw: *"Treated
in this project as signal (coordinated abuse), not an error to fix, but
'cluster' here is not a verified single-person identity."* The honest,
carefully-scoped claim this project makes is never "we identify individual
people" — it's "we group transactions that plausibly share an underlying
actor, in a way that's internally label-consistent 97.61% of the time, and
we say so explicitly whenever that grouping is presented to a human."

---

# PART 4 — GRAPHS

## Graph theory from zero

A **graph**, in the mathematical sense used here, is nothing more exotic
than a collection of "things" and a collection of "connections between
pairs of things." The formal names:

- A **node** (also called a *vertex*) is one "thing." In this project, a
  node is a uid — one resolved entity.
- An **edge** is a connection between exactly two nodes, meaning "these two
  things are linked." In this project, an edge between two uids means
  "these two uids share some strong identifying signal" (the exact rules
  are in the next section).
- The **degree** of a node is simply how many edges touch it — how many
  other nodes it's directly connected to. A node with degree 0 is
  **isolated** — linked to no one.
- A **connected component** is a maximal group of nodes where you can get
  from any one node to any other by following edges (possibly through
  several hops), and no node in the group has any edge reaching outside
  it. Picture three friends who've all transacted from the same email
  domain and address — even if uid A only directly links to uid B, and
  uid B links to uid C, but A and C never directly linked to each other, A,
  B, and C still form one connected component, because you can walk from A
  to C via B. A node with degree 0 is its own connected component, all by
  itself — a "cluster of one."
- **Density** of a component measures how thoroughly-connected its members
  are, relative to the maximum possible: if a component has `V` nodes, the
  maximum possible number of edges (every node connected to every other) is
  `V * (V-1) / 2`. Density is `(actual edges) / (maximum possible edges)`.
  A density of 1.0 means every member is directly linked to every other
  member (a **clique**). A density near 0 means the component is
  connected, but only loosely — perhaps just a single chain linking
  everyone.
- **k-core** is a way of measuring how *deeply embedded* a node is in a
  dense region, beyond just density or raw degree. A graph's "k-core" is
  the largest sub-part of it where every remaining node has degree at
  least `k` *within that sub-part* (after repeatedly stripping away nodes
  with fewer than `k` neighbors, whatever's left, if anything, is the
  k-core). A node's own **core number** is the largest `k` for which it
  still belongs to some k-core. Picture a tree shape — a hub with several
  leaves hanging off it, and nothing else. Strip away every node with
  degree less than 2 (every leaf has degree 1), and the hub itself now has
  no neighbors left either — the whole thing collapses to nothing. A tree
  can never have a core number above 1, no matter how many leaves it has,
  because it has no *cycle* (no way to come back around) to sustain a
  denser sub-region. A tightly mutually-connected clique, by contrast, has
  every node holding up every other node — a 4-node clique (`K4`, where
  every one of the 4 nodes connects to all 3 others) is its own 3-core,
  because stripping anything with degree less than 3 removes nothing at
  all — every node already has degree exactly 3.

This last distinction — a star/tree shape vs. a clique — matters
specifically because two very differently-*shaped* clusters can look
identical by simpler measures. This project's own test
(`tests/test_graph.py`) constructs exactly this case to prove the point:

```python
def _topology_frame():
    """Three clusters chosen to need BOTH new features to tell apart:
    a star (hub uidH + 3 leaves, a tree -- no cycle) and a same-size clique
    (uidP/Q/R/S, K4) end up with the identical star_ratio (max degree /
    cluster size), by construction -- that's the whole reason star_ratio
    alone doesn't distinguish shape and has to be read alongside
    cluster_edge_density (already existing) and k_core_number (new)...
    """
```

and the corresponding assertions:

```python
    # Star (hub degree 3, cluster size 4): star_ratio = 3/4. A tree has no
    # cycle, so k-core tops out at 1 for every connected node in it.
    assert hub["star_ratio"] == pytest.approx(0.75)
    assert hub["k_core_number"] == 1
    assert hub["cluster_edge_density"] == pytest.approx(3 / 6)  # 3 edges / C(4,2)

    # Clique (K4): every node degree 3, same cluster size 4 -> the SAME
    # star_ratio as the star above -- demonstrating why star_ratio alone
    # can't tell the two shapes apart.
    assert clique_member["star_ratio"] == pytest.approx(0.75)
    assert clique_member["cluster_edge_density"] == pytest.approx(1.0)  # 6/6
    assert clique_member["k_core_number"] == 3
```

A hub-and-spoke shape (one node linked to three others, who aren't linked
to each other — think one device used to register several accounts) and a
tight mutual clique of the same size (four accounts all directly sharing
signals with each other — think a small, deliberate ring) produce the
*identical* `star_ratio` (0.75, since the formula is "highest single node's
degree, divided by cluster size," and both shapes happen to have a max
degree of 3 among 4 members) — but a completely different density (0.5 vs.
1.0) and a completely different core number (1 vs. 3). This is exactly why
this project's dashboard describes edge density, star ratio, and k-core as
features that must be read *together*, not any one alone.

## Each linkage rule, and why it was chosen

Two uids get an edge in this project's entity graph if they share a value
on any one of three rules, defined in `src/graph.py`:

```python
LINKAGE_RULES: tuple[LinkageRule, ...] = (
    LinkageRule(
        name="device_info",
        key_columns=("DeviceInfo",),
        rationale=(
            "The exact same device fingerprint appearing across multiple "
            "client identities is strong evidence of a shared physical "
            "device -- one operator running several card+address "
            "combinations, or a device farm."
        ),
    ),
    LinkageRule(
        name="addr1_email",
        key_columns=("addr1", "P_emaildomain"),
        rationale=(
            "A shared delivery/billing address code AND a shared purchaser "
            "email domain together are a stronger joint signal than either "
            "alone: many people share a common email provider, and many "
            "share a common region-coded addr1 value, but the co-occurrence "
            "of both narrows this considerably."
        ),
    ),
    LinkageRule(
        name="card_bank_addr",
        key_columns=("card3", "card5", "addr1"),
        rationale=(
            "card3/card5 are issuing-bank/network category codes that are "
            "essentially fixed once card1 is fixed (confirmed empirically "
            "in the D1 investigation -- they never varied across the 20 "
            "largest uids). Requiring them to match jointly WITH addr1 "
            "links different card1 values that share the same "
            "issuer/network profile and address -- catching a coordinated "
            "actor issuing multiple card numbers to the same address."
        ),
    ),
)
```

Notice the shared logic across all three: none of them link on a *single*,
weak field alone. `addr1_email` requires *both* address and email domain to
match together (either alone is too common to mean anything by itself —
verified in this project's own test, `test_addr1_email_requires_both_to_match`,
which confirms two uids sharing an address but *not* an email domain do
*not* get linked). `card_bank_addr` requires *three* fields together. Only
`device_info` links on one field alone, and that's specifically because a
matching exact device fingerprint string is already an unusually specific
signal on its own — two different card+address combinations transacting
from the literal same device string is a strong, specific coincidence,
unlike sharing a common email provider or a common region code.

Real proof this specificity matters, from `results/case_studies.md`'s
manually-inspected top-priority cluster: cluster case 2's two members were
linked purely by `device_info`, and the shared value wasn't a generic OS
name — it was `SM-G950F Build/NRD90M`, a precise Samsung Galaxy S8
model-and-build identifier. The case study calls this out directly:
*"this is the most specific evidence of the three cases — a precise device
build string, not a locale or OS name, linking two otherwise-unconnected
card+address combinations."* Contrast with case 3 in the same report, where
two uids were linked by `device_info` too, but the shared value was
`"en-gb"` — a locale string, not a device fingerprint — and the report
flags this explicitly as the weakest of the three cases: *"'en-gb' reads as
a locale/language string, not a device fingerprint — it's the kind of
value real, unrelated UK-locale users could plausibly share by chance."*
The linkage *rule* is the same in both cases; the *specificity of the value
that satisfied it* is what actually determines how much to trust the
resulting edge — and this project's own case-study write-up says so
plainly rather than treating every edge as equally strong.

## The hub problem: why max_degree=1000 collapses 64% of uids into one component

Not every shared value is meaningful evidence of a relationship. If ten
thousand unrelated people all happen to use `gmail.com` and live in the
same broad region code, that shared value tells you nothing about any
*specific* pair among them — it's just a common default, not a
relationship. `build_entity_graph`'s **hub guard** handles this: any value
shared by more than `max_degree` distinct uids is excluded from linkage
entirely, and the exclusion is logged, not silently dropped.

The function's own built-in default for `max_degree` is 1000 — but this
project discovered, by actually sweeping the parameter against the real
data (not by guessing), that this default is unusable on this dataset. The
exact numbers, from `src/graph.py`'s own module docstring and confirmed in
`ARCHITECTURE.md`:

| max_degree | largest resulting component | % of all uids |
|---:|---:|---:|
| 20 | 126 uids | 0.06% |
| 30 | 919 uids | 0.46% |
| 35 | 5,141 uids | 2.58% |
| 1000 (the function's own default) | **127,708 uids** | **64%** |

Read that middle row carefully: moving `max_degree` from 30 to 35 — a small
change — jumps the largest component from 919 uids to 5,141, more than a
five-fold increase from one small parameter step. This is a **phase
transition**, not a smooth curve: below some threshold, hub-like values
stay excluded and the graph stays sparse and meaningful; cross that
threshold and one or two extremely common values (a dominant `addr1`+`gmail.com`
combination, or the single most common card-network profile) start acting
as bridges that fuse thousands of otherwise-unrelated identities into one
giant blob. At the literal default of 1000, nearly two-thirds of the
*entire dataset's* resolved identities end up in one single connected
component — at which point every cluster-level feature computed from graph
structure (cluster size, density, and so on) becomes nearly constant across
64% of the population, because they're all, structurally, "the same
cluster." That destroys the entire signal this project is trying to
extract.

This project's pipeline actually uses `max_degree=20`
(`run_pipeline.MAX_DEGREE = 20`), which keeps the largest real cluster at
just 126 uids (0.06% of the population) — deliberately conservative. The
trade-off, stated explicitly in `ARCHITECTURE.md`: *"a higher max_degree
would catch a few more genuine large rings, but the real cost is a single
supercluster that makes cluster-level features nearly constant across the
whole population — worse than missing some larger rings."* Concretely,
this excludes 1,114 distinct values, covering 376,264 uid-appearances in
total (`results/ablation.md`'s Graph construction section) — the ten
largest excluded values include generic device strings (`Windows`: 18,535
uid-appearances; `iOS Device`: 12,719; `MacOS`: 8,344) and the single most
common card-network/address and address/email combinations. None of these
are wrong to exclude — they are, definitionally, too common to be evidence
of anything specific — but the *threshold* for "too common" is a judgment
call this project made deliberately conservative, and says so, rather than
tuning it after the fact to produce a nicer-looking result.

---

# PART 5 — CAUSAL FEATURES AND TEMPORAL VALIDATION

## What temporal leakage is, with a concrete inflation example

**Temporal leakage** is a specific form of the leakage introduced in Part
1.6: information from *after* the moment a real decision would need to be
made sneaks into a feature or a split, in a system whose entire purpose is
predicting the future from the past. It's easiest to see with a concrete,
hypothetical (not this project's actual number — illustrative) example.

Imagine computing a feature called "has this cluster ever had a fraud
transaction" by looking at *all* of a cluster's transactions, regardless of
date, and using that feature to score a transaction from January. Suppose
this cluster's very first fraud transaction actually happens in June. If
the "ever had fraud" feature is computed over the whole dataset without
regard to time, the January transaction gets marked "this cluster has had
fraud" — using a fact that wouldn't be knowable until June. A model trained
on this leaked feature will look spectacular: it will have learned to
essentially read off the answer for every transaction belonging to a
cluster that *eventually* gets caught, rather than learning to *predict*
which clusters will turn out to be fraudulent before they're caught. Report
that model's test-set metrics, and you get an inflated, unrealistic number
that would collapse the moment the system ran live in production — where,
by definition, June's information does not exist yet when you're scoring a
January transaction.

This is exactly the failure mode this project's dominant feature,
`cluster_prior_fraud_share`, is at the highest risk of falling into — it is
*precisely* a "has this cluster had fraud before" feature, which is why
Part 5's remaining sections, and this project's own sanity checks, exist:
to prove, not just claim, that this particular danger was actually avoided.

## Why random train/test splits are invalid on time-ordered data

A standard machine learning workflow randomly shuffles all rows, then
splits some fraction into a test set. This is invalid here for a reason
directly connected to Part 1.6 and the leakage risk above: this project's
features are built by aggregating a *cluster's* transactions together, and
a real-world cluster's transactions are naturally spread out over time. If
rows are shuffled randomly before splitting, a single cluster's earlier and
later transactions get scattered arbitrarily across both the train and test
sets. Train a model, and the test set would contain transactions whose
cluster's *other* transactions — including ones that happened *later* in
real time — were fully visible during training. That's temporal leakage by
construction, baked into the split itself, before a single feature is even
computed.

`CLAUDE.md`'s rule is unambiguous: *"Splits are ALWAYS temporal on
TransactionDT. Never random."* The actual split logic, `src/evaluate.py`'s
`temporal_train_test_split` (frozen — this file can never be edited to
improve reported numbers, per the same rules file):

```python
def temporal_train_test_split(
    transactions: pd.DataFrame,
    dt_col: str = "TransactionDT",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = transactions.sort_values(dt_col, kind="mergesort")
    n = len(sorted_df)
    target_test_n = round(n * test_size)

    counts_by_dt = sorted_df[dt_col].value_counts()
    unique_dts = np.sort(sorted_df[dt_col].unique())

    cumulative = 0
    split_dt = unique_dts[-1]
    for dt in unique_dts[::-1]:
        cumulative += counts_by_dt[dt]
        split_dt = dt
        if cumulative >= target_test_n:
            break

    train = sorted_df[sorted_df[dt_col] < split_dt]
    test = sorted_df[sorted_df[dt_col] >= split_dt]

    if len(train) and len(test):
        assert train[dt_col].max() < test[dt_col].min()

    return train, test
```

Walking through this mechanically: sort every row by timestamp. Walk
backward from the most recent timestamp, accumulating how many rows share
each exact timestamp, until the accumulated count reaches the requested
test fraction (20% by default). Everything strictly before that boundary
timestamp becomes `train`; everything at or after it becomes `test`. The
walk is careful to stop only at a *whole* timestamp boundary — never
splitting a group of rows that share the exact same `TransactionDT` value
across train and test — because doing so would put some same-instant
transactions on each side of the boundary, a small but real edge case of
the same leakage problem. This is why the realized test fraction can drift
slightly from the requested 20%: on this project's actual 80% split, it
landed at 118,108 test rows out of 590,540 total (472,432 train), a hair
different from an exact 20% (118,108) because of exactly this
tie-respecting logic — an `assert` right in the function itself,
`train[dt_col].max() < test[dt_col].min()`, checks that every single train
timestamp is strictly earlier than every single test timestamp, every time
this function runs, on any data.

## Exactly what as_of does, mechanically

`as_of` is the timestamp boundary — the first moment in the test period —
passed into `graph.compute_cluster_features` to enforce causality at the
*feature* level, not just at the split level. From `run_pipeline.py`:

```python
train_df, test_df = evaluate.temporal_train_test_split(df)
as_of = float(test_df["TransactionDT"].min())

entity_graph = graph.build_entity_graph(train_df, entity_ids, max_degree=MAX_DEGREE)
cluster_features = graph.compute_cluster_features(
    train_df, entity_ids, entity_graph.graph, as_of=as_of
)
```

`as_of` is simply the earliest `TransactionDT` in the test set — the exact
moment the "future" begins, from the model's point of view. It's passed
into `compute_cluster_features`, which enforces it internally, at the very
top of the function, before any aggregation happens:

```python
def compute_cluster_features(
    transactions: pd.DataFrame,
    entity_ids: pd.Series,
    graph: nx.Graph,
    as_of: float | None = None,
    include_topology: bool = False,
) -> pd.DataFrame:
    txns = _prepare(transactions, entity_ids)
    if as_of is not None:
        txns = txns.loc[txns["TransactionDT"] < as_of]
    ...
```

Every single downstream aggregate in this function — cluster size,
transaction count, edge density, velocity, amount variability, burst
concentration, email diversity, and critically `cluster_prior_fraud_share`
— is computed from `txns` *after* this filter has already run. There is no
code path later in the function that goes back and reads from the original,
unfiltered `transactions` argument. This is what makes the causal guarantee
mechanical rather than a matter of trusting that callers remembered to
filter first: even if a caller passed in the full, unfiltered dataset
(train and test both), `compute_cluster_features` would still only ever
compute from the pre-`as_of` slice, because it does its own filtering
internally.

There's a second, separate causal guarantee this project relies on for the
*graph structure itself* (which uids are linked to which — as opposed to
the transaction-level aggregates above): `build_entity_graph` is called on
`train_df` only, never on the full dataset. `test_df` is never passed to
it. This means no test-period transaction can ever contribute a node or an
edge to the graph in the first place — a structural guarantee, not a
runtime filter, since the function is simply never given the opportunity to
see test-period data at all.

## How the leakage test works, and what it would catch

This project doesn't just assert causal correctness in a docstring — it
checks it two independent ways: a unit test with adversarially-chosen
inputs, and a full sweep against the real dataset.

The unit test, `tests/test_graph.py`'s
`test_compute_cluster_features_ignores_transactions_at_or_after_as_of`,
constructs a transaction specifically designed to move every single
feature if it leaked in — an enormous amount, a fraud label, and a brand
new email domain, dated after `as_of`:

```python
def test_compute_cluster_features_ignores_transactions_at_or_after_as_of():
    df, entity_ids, graph = _feature_frame()
    as_of = 2000

    future_row = pd.DataFrame(
        {
            "TransactionID": [99],
            "TransactionDT": [5000],  # after as_of
            "TransactionAmt": [999999.0],
            "P_emaildomain": ["future-only-domain.com"],
            "isFraud": [1],
        }
    )
    future_entity = _entity_ids([99], ["uidA"])

    df_with_future = pd.concat([df, future_row], ignore_index=True)
    entity_ids_with_future = pd.concat([entity_ids, future_entity])

    with_future = compute_cluster_features(
        df_with_future, entity_ids_with_future, graph, as_of=as_of
    )
    without_future = compute_cluster_features(df, entity_ids, graph, as_of=as_of)

    pd.testing.assert_frame_equal(
        with_future.loc[["uidA", "uidB", "uidC"]].sort_index(),
        without_future.loc[["uidA", "uidB", "uidC"]].sort_index(),
    )
```

The logic: compute the features twice, once on a dataset with this
"planted" future transaction and once without it, and assert the resulting
feature tables are byte-for-byte identical for every existing uid. If any
line of `compute_cluster_features` accidentally read from a row that should
have been filtered out, at least one of `cluster_amt_cv` (the amount would
shift the variance), `cluster_prior_fraud_share` (the fraud flag would
raise it), `uid_email_domain_count` (a new domain would raise it), or
`cluster_txn_count`/`cluster_velocity` (an extra transaction would raise
both) would differ between the two runs, and the test would fail
immediately. The planted row's values were chosen specifically to be
impossible to miss if leaked — a huge, distinctive, guaranteed-to-move-
every-metric outlier, not a subtle edge case that could slip through by
coincidence.

The second check runs the same logic against the real, full dataset, not a
synthetic fixture — described in `results/ablation.md`'s Sanity checks
section: *"Checked all 155,579 clusters that have both a reported
cluster_prior_fraud_share and an independently-recomputable pre-as_of
value: comparing the pipeline's reported value against one computed
straight from raw rows with TransactionDT < as_of, bypassing graph.py
entirely. Mismatches: 0."* This is a genuinely independent recomputation —
a second implementation of the same logic, written separately, that
bypasses `graph.py`'s code path entirely and recomputes the same quantity
straight from raw transaction rows — checked against every single one of
the real dataset's 155,579 qualifying clusters, not a sample. Zero
mismatches. The report goes further and finds a concrete, real example
where the *would-leak* value differs from the *correctly-filtered* value —
cluster #17894, whose fraud-labeled transactions both fall in the test
period: the correctly-filtered `cluster_prior_fraud_share` reports
**0.0000** (correct — no fraud had happened yet, as of the train/test
boundary), while the naive all-time version (ignoring the as_of cutoff
entirely) would have reported **1.0000** for the exact same cluster. This
is the leakage test doing real, demonstrable work — not a check that
happens to pass vacuously because nothing in the real data would ever
trigger it.

---

# PART 6 — THE ABLATION

## What an ablation is, and why it isolates an effect

An **ablation study** (the term comes from surgery — removing a part of
something to see what changes) is a comparison between two systems that are
identical in every respect except one, specifically so that any measured
difference in outcome can be attributed to that one difference alone. If
the two systems being compared differ in more than one way, an observed gap
between them is ambiguous — you don't know which of the differences
actually caused it, or whether they interacted. The scientific point of an
ablation is to eliminate every variable except the one under test, so the
result is a clean, attributable measurement rather than a confounded one.

This project's central ablation compares a **baseline model** (trained on
transaction features only) against a **cluster model** (the identical
transaction features, plus the ten engineered cluster features from Part
4). Everything else — training procedure, hyperparameters, random seed,
number of boosting rounds, evaluation code, test set — is held byte-for-byte
identical between the two, which is exactly why Part 1.2 quoted
`model.py`'s own docstring insisting that both models call the *same*
private `_fit()` function. If the two training paths had diverged in any
other way — different hyperparameters, a different random seed, a
different number of trees — a measured gap in PR-AUC between them would be
ambiguous: is it the cluster features, or is it some unrelated difference in
how the two models were trained? By making everything else identical, any
measured difference can only be attributed to the one thing that actually
differs: the presence of the ten cluster-derived columns.

This project's own test enforces this structurally, not just by
convention — `tests/test_model.py`'s
`test_baseline_and_cluster_training_use_identical_params_and_seed`:

```python
def test_baseline_and_cluster_training_use_identical_params_and_seed():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)})
    y = pd.Series((X["f1"] > 0.5).astype(int))

    baseline_model = train_baseline_model(X, y)
    cluster_model = train_cluster_model(X, y)  # same X on purpose here

    # Identical training paths on identical data must produce identical
    # trees -- this is the ablation's entire premise (same seed, same
    # params, same boosting rounds).
    assert baseline_model.model_to_string() == cluster_model.model_to_string()
```

Given the exact same input data, `train_baseline_model` and
`train_cluster_model` must produce *byte-identical* trained models
(`model_to_string()` serializes the entire tree structure to text) — proof
that the two training functions really do share every mechanical detail,
and the *only* place they can possibly diverge is which columns end up in
`X` when they're called with genuinely different feature sets.

## Every metric in the results table

Here is the project's headline table, verbatim from `results/ablation.md`:

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |
|---|---:|---:|---:|
| baseline | 0.5646 | 0.4791 | 30078.40 |
| cluster | 0.6322 | 0.5576 | 26155.72 |
| cluster (no `cluster_prior_fraud_share`) | 0.5756 | 0.4951 | 29317.66 |

**PR-AUC** (defined fully in Part 1.5) — area under the precision-recall
curve; 1.0 is perfect ranking, ~0.035 (the base fraud rate) is random
guessing. Computed by `src/evaluate.py`'s `pr_auc`, a one-line wrapper
around scikit-learn's `average_precision_score`. Good values for this task,
given the base rate, sit meaningfully above 0.035; the cluster model's
0.6322 is roughly 18 times the random-guessing floor.

**Recall @ 1% FPR** (also defined in Part 1.5) — of all real fraud, what
fraction does the model catch, if you're only willing to tolerate a 1%
false-positive rate among legitimate transactions? This is a specific,
realistic **operating point**: 1% FPR at this dataset's ~3.5% base rate
means disturbing a genuinely small slice of legitimate traffic while still
trying to catch as much fraud as possible — a meaningful, low-friction
point to measure, not an arbitrary choice. Computed by `recall_at_fpr`:

```python
def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    achievable = tpr[fpr <= target_fpr]
    if achievable.size == 0:
        return 0.0
    return float(achievable.max())
```

This sweeps every possible threshold (via scikit-learn's `roc_curve`,
which returns every FPR/TPR pair as the threshold varies), keeps only the
thresholds whose FPR doesn't exceed the 1% target, and reports the best
(highest) recall achievable among those. The baseline model achieves
0.4791 (catches 47.91% of fraud at a 1% false-positive budget); the
cluster model achieves 0.5576 — a real, meaningful lift at this
specific, realistic operating point.

**Cost per 10k txns** is this project's attempt to translate the abstract
precision/recall trade-off into a single business-relevant number: the
expected dollar-equivalent cost, per 10,000 transactions processed, of the
model's mistakes at a chosen threshold. From `src/evaluate.py`:

```python
def cost_per_10k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    predicted_positive = y_score >= threshold

    fn = np.sum((y_true == 1) & ~predicted_positive)
    fp = np.sum((y_true == 0) & predicted_positive)
    n = len(y_true)

    return float((fn * cost_fn + fp * cost_fp) / n * 10000)
```

Count every **false negative** (real fraud the model missed, weighted by
`cost_fn`, the assumed dollar cost of a missed fraud case) and every
**false positive** (a legitimate transaction wrongly flagged, weighted by
`cost_fp`, the assumed cost of an unnecessary friction event), sum the
weighted costs, divide by the total number of transactions scored, and
scale up to a "per 10,000 transactions" rate — a normalized number that
doesn't depend on how large the specific test set happens to be, so it's
comparable across different-sized evaluations. Crucially, `cost_fn` and
`cost_fp` are *parameters*, never hardcoded inside the function — a
project-wide discipline, checked directly by
`tests/test_evaluate.py`'s `test_cost_per_10k_is_parameterised_not_hardcoded`,
which asserts that changing the cost ratio actually changes the reported
cost.

The specific costs used to produce the 30078.40/26155.72/29317.66 figures
above are `cost_fn=500.0` and `cost_fp=5.0` — a **100:1 ratio**, stated
explicitly and repeatedly throughout this project's results as
**illustrative, not a Razorpay figure**: *"cost assumptions are
illustrative, NOT Razorpay figures: cost_fn=500 (a missed abuse case),
cost_fp=5 (a false alarm / unnecessary step-up) — a 100:1 ratio, chosen to
represent a chargeback loss being much costlier than customer friction,
nothing more precise than that"* (`results/ablation.md`). This ratio drives
everything about where the cost-minimizing threshold ends up (Part 9) — a
different, less aggressive ratio would move that threshold substantially,
and the resulting numbers with it.

## What the ablation actually found — reading the table honestly

The cluster model beats the baseline on all three metrics: PR-AUC **+0.0676**
(0.6322 vs. 0.5646), recall@1%FPR **+0.0785** (0.5576 vs. 0.4791), cost per
10k **−3922.68** (26155.72 vs. 30078.40, negative meaning cheaper — an
improvement). That's the headline, and it's real.

But this project doesn't stop at the headline — it interrogates its own
strongest number, and finds that almost all of it comes from one feature.
The third table row, "cluster (no `cluster_prior_fraud_share`)," retrains
the exact same cluster model with just one column deleted —
`cluster_prior_fraud_share`, the "share of this cluster's members who have
ever had a fraud transaction before" feature — and PR-AUC drops from 0.6322
almost all the way back down to 0.5756, barely above the 0.5646 baseline.
Arithmetically: the full lift is 0.6322 − 0.5646 = **+0.0676**; with that
one feature removed, the remaining lift is only 0.5756 − 0.5646 = **+0.0110**.
That means (0.0676 − 0.0110) / 0.0676 ≈ **84%** of the entire measured
improvement comes from a single engineered feature — a number this
project's own report states directly: *"About 84% of the headline PR-AUC
lift (+0.0676) comes from a single feature, cluster_prior_fraud_share."*

Why this one feature dominates so completely is confirmed by the model's
own feature-importance ranking (`results/ablation.md`, "gain" = how much a
feature reduced the training loss across every tree that used it, summed
over all 300 boosting rounds):

| feature | gain |
|---|---:|
| cluster_prior_fraud_share | 558,334.3 |
| V258 | 63,550.3 |
| C1 | 52,819.7 |
| DeviceInfo | 28,908.5 |
| C14 | 27,892.7 |

`cluster_prior_fraud_share`'s gain is nearly **9x** the next-highest
feature (`V258`, one of Vesta's own anonymized engineered signals from Part
2). This single feature isn't just contributing meaningfully — it
overwhelmingly dominates the entire trained model. Whether that's a sign of
genuine, powerful signal or a red flag worth investigating is exactly the
question Part 5's sanity checks and Part 7's stability analysis exist to
answer.

---

# PART 7 — THE STABILITY FINDING

## What was tested, and what a sign flip means statistically

A single train/test split is one sample from a range of possible outcomes
— the specific 80%-through-time boundary this project's headline numbers
use is one particular cut of the calendar, and a different cut might tell a
somewhat different story purely by chance, independent of whether the
underlying features are genuinely useful. To find out whether the headline
+0.0676 PR-AUC lift is a stable property of the feature set, or an artifact
of that one specific boundary, this project re-ran the *entire* ablation
(fresh entity graph, fresh causal cluster features, fresh baseline/cluster/
trimmed models — nothing cached or reused) at four different rolling split
points: 60%, 70%, 80%, and 90% of the way through sorted `TransactionDT`.

The full results, from `results/stability.md`:

| split | baseline PR-AUC | cluster PR-AUC | full lift | trimmed PR-AUC (no `cluster_prior_fraud_share`) | trimmed lift |
|---:|---:|---:|---:|---:|---:|
| 60% | 0.5620 | 0.5903 | +0.0284 | 0.5576 | **−0.0044** |
| 70% | 0.5570 | 0.6210 | +0.0641 | 0.5756 | **+0.0186** |
| 80% | 0.5646 | 0.6322 | +0.0676 | 0.5756 | **+0.0110** |
| 90% | 0.6243 | 0.6706 | +0.0463 | 0.6231 | **−0.0013** |

Two separate findings live in this one table. First: the **full** lift
(cluster model vs. baseline, with every feature including
`cluster_prior_fraud_share`) is positive at all four splits — mean
**+0.0516**, but with a **spread** (max minus min) of **0.0393**, which is
a large fraction of the mean itself. In plain terms: the cluster features
reliably help *some* amount at every split tested, but exactly how much
varies substantially — anywhere from +0.0284 to +0.0676 depending on which
6-month window you happen to test on. The single-split headline number
(+0.0676) turns out to be close to the *best* case among the four splits
measured, not a typical one.

Second, and far more consequential: the **trimmed** lift (what's left once
`cluster_prior_fraud_share` is removed — the "genuine structural signal"
this project originally hoped the graph features would independently
provide) is **positive at 70% and 80%, but negative at 60% and 90%.** This
is a **sign flip** — the measured effect isn't just noisy in *magnitude*,
it flips between "the structural features help" and "the structural
features actually hurt slightly," depending on which of four tested
windows you happen to look at. Statistically, this is close to the
strongest possible signal that an effect is not real, or at least not
reliably measurable at this sample size: an effect whose sign depends on
which subset of the data you happened to test on is indistinguishable from
an effect that's actually zero, buried under sampling noise that happens to
occasionally push it one way or the other. The mean across all four splits
is **+0.0060** — barely different from zero, with a spread of 0.0230 that's
nearly four times the mean itself.

## The topology follow-up, and its negative result

Since the aggregate structural features (cluster size, edge density,
velocity, amount variability, burst concentration, email diversity) didn't
show a reliable lift, this project tried a second, more sophisticated
attempt: two genuine **topology** features (introduced in Part 4 —
`k_core_number`, how deeply embedded a node is in a dense sub-region, and
`star_ratio`, how hub-and-spoke-shaped a cluster is) that capture cluster
*shape* rather than just size or density aggregates. The hypothesis: maybe
the aggregates were simply too crude to capture a real shape-based abuse
signal (a device farm's hub-and-spoke pattern vs. a tight, mutually-linked
ring) that richer topology features could find.

On the single 80% split, adding topology to the trimmed feature set moved
PR-AUC from 0.5756 to 0.5711 — a **small decrease**, not an improvement
(`results/ablation_topology.md`). Feature importance confirms these two new
features contributed almost nothing to the trained model: out of 443 total
features, `k_core_number` ranked **228th** (gain 187.0) and `star_ratio`
ranked **113th** (gain 948.0) — present, technically used by some trees,
but nowhere near meaningful.

The same four-split stability test was then repeated with topology
included, to check whether it at least stabilized the *sign* of the
residual lift even if it didn't move the single-split number much. It
didn't — `results/stability_topology.md`:

| split | trimmed lift (no topology) | trimmed lift (WITH topology) |
|---:|---:|---:|
| 60% | −0.0044 | **−0.0020** |
| 70% | +0.0186 | **+0.0135** |
| 80% | +0.0110 | **+0.0065** |
| 90% | −0.0013 | **−0.0104** |

Still negative at 60% and 90%, still positive at 70% and 80% — the exact
same sign pattern as without topology, just with a slightly lower mean
(+0.0019 vs. +0.0060) and almost the same spread (0.0239 vs. 0.0230). The
report's own verdict is unambiguous: *"Topology does not rescue the
residual lift... neither version shows a reliable, sign-stable lift once
the dominant cluster_prior_fraud_share confound is removed, on this
dataset at this sample size."*

## What was corrected in the README, and why that's this project's strongest feature

Before this stability analysis existed, this project's own README described
the non-`cluster_prior_fraud_share` structural features as contributing "a
smaller but genuine residual lift" — a claim that felt reasonable looking
at the single 80% split alone (+0.0110 does look like a small, real,
positive number, in isolation). The stability analysis showed that claim
doesn't survive contact with a second data point, let alone four. The
project's own correction, stated in `results/ablation.md`, doesn't soften
this finding or bury it — it states it more strongly than before:

> "**Correction, not a softening:** this project previously described the
> residual lift once cluster_prior_fraud_share is removed as 'a smaller but
> genuine residual lift' from the remaining structural/graph features...
> That claim does not hold up. Across the same 4 splits, the residual lift
> is mean +0.0060, spread 0.0230, and changes sign... Graph-structure
> features alone do not show a reliable lift on this dataset at this
> sample size — only cluster_prior_fraud_share shows a consistently
> positive effect across splits, and per README.md's Limitations note,
> that effect is itself partly circular."

`README.md`'s current Limitations section reflects this directly, and was
strengthened (not softened) a second time after the topology follow-up
confirmed the same conclusion:

> "**The dominant feature is backward-looking, partly circular fraud
> history — and richer topology features don't change that.**
> `cluster_prior_fraud_share` measures label propagation across a card, not
> independently-discovered abuse, and carries essentially all of the
> reliably measured lift; a second attempt with topology features (k-core
> depth, hub-vs-clique shape) still didn't produce a stable effect across
> splits, which is a finding about this dataset and sample size, not a
> bug."

This — running a second, harder version of the same test after the first
one came back negative, then updating the project's own headline claims to
match what was actually found, in both directions (strengthening a
limitation, not softening it) — is this project's strongest single
methodological feature, and worth stating why explicitly for an interview:
it demonstrates the single most important discipline in applied ML work —
being more skeptical of your own best-looking number than of a
disappointing one, and being willing to publicly revise a claim once
better evidence contradicts it, rather than quietly leaving the flattering
version standing. Anyone can report a good number. Re-testing a good
number under harder conditions, finding it doesn't hold up, and saying so
in the same document that reports the original number, is a different and
much rarer discipline — and it is exactly what an interviewer evaluating
research maturity, as opposed to just result-reporting, is listening for.

---

# PART 8 — THE LLM AND POLICY LAYERS

## Why the AI does not make the decision, argued properly

This project has two entirely separate layers downstream of the trained
model, and keeping them separate is treated as a hard architectural rule,
not a soft preference: `src/policy.py` is a **deterministic decision
layer** — its only job is comparing a model score against two fixed
thresholds and returning one of three actions (`allow`, `step_up`,
`review`). `src/investigator.py` is an **LLM (large language model)
layer** — its only job is producing a human-readable narrative explanation
and a relative priority ranking for a cluster, for a human investigator to
read. `CLAUDE.md`'s rule: *"The LLM layer explains and prioritizes.
policy.py decides. Never merge them."*

The argument for *why* this separation matters, made properly rather than
just asserted: an LLM is a language model — it is very good at producing
fluent, plausible-sounding text conditioned on the input it's given, but
it has no guarantee of numerical consistency, no guarantee of
determinism (the same input can, in principle, produce a differently-worded
output on a different call), and — most importantly for a compliance-
sensitive payments decision — no auditable, reproducible chain of
reasoning that a regulator or an internal audit process can mechanically
re-verify after the fact. A deterministic score-vs-threshold comparison,
by contrast, can be re-run byte-for-byte identically a year later, given
the same score and the same threshold, and will always produce the same
answer — which is exactly the property you want underpinning an action
that affects a real customer's ability to transact. Using an LLM's
narrative *quality* (however good) as an input into *which action gets
taken* would mean a customer's outcome could, in principle, depend on
wording variance in a generated paragraph — an unacceptable and
unauditable basis for an automated financial decision. Keeping the LLM
strictly downstream of the decision — explaining a decision that was
already made deterministically — means the LLM's occasional imperfection
(explored below) can never itself become the thing that determines whether
a real transaction gets blocked.

This separation is enforced *structurally*, not just documented, and
checked two independent ways by `tests/test_policy.py`. First, statically —
parsing `policy.py`'s own source code and confirming it contains no import
of `investigator` anywhere, at all:

```python
def test_policy_module_does_not_import_investigator():
    source = Path(policy.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("investigator" in name for name in imported)
```

This uses Python's own `ast` (abstract syntax tree) module to parse
`policy.py`'s source code into a structured representation and walk every
import statement in it — a check that can't be fooled by a comment or a
docstring merely *claiming* no dependency exists; it inspects the actual
parsed code. Second, behaviorally — proving the *decisions themselves*
don't change whether or not the investigator layer (and by extension, any
LLM call) is even available:

```python
def test_decisions_identical_with_investigator_disabled(monkeypatch):
    scores = pd.DataFrame(
        {"score": [0.001, 0.02, 0.5]}, index=["u1", "u2", "u3"]
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    result_enabled = policy.apply_policy(scores)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "src.investigator", None)  # simulate unavailable
    result_disabled = policy.apply_policy(scores)

    pd.testing.assert_frame_equal(result_enabled, result_disabled)
```

Run `apply_policy` once with an API key present and the investigator
module fully importable, and once with the key removed and the module
itself simulated as unavailable (`sys.modules["src.investigator"] = None`)
— and assert the resulting decisions are identical, row for row. If
`policy.py` ever accidentally came to depend on `investigator.py` in any
way, this test would catch it by making the LLM layer's absence
observable in the decision output — and it isn't.

## How groundedness was measured, and what it proves

An LLM writing a narrative about a cluster's evidence could, in principle,
hallucinate — state a number that sounds plausible but doesn't actually
appear anywhere in the real evidence it was given. This project calls a
narrative **grounded** if every number it states traces back to a real
value in its evidence input, and built a specific, checkable rule to
enforce this rather than hoping for it. The system prompt given to the LLM
(`src/investigator.py`) states the rule directly, in the strongest terms
the prompt uses for anything:

> "HARD RULE, more important than anything else in this prompt: you must
> NEVER state a number, percentage, count, or statistic that is not
> present verbatim (or trivially rounded, e.g. 0.78 from 0.7797) in the
> evidence JSON below... This is checked programmatically after you
> respond; an invented number is a failure of this task, not a minor style
> issue."

The checking logic (`run_pipeline.py`'s `_ungrounded_claims`) extracts
every number that appears anywhere in the generated narrative text with a
regular expression, then checks each one against every value in the
evidence dictionary that was actually given to the model — allowing for
reasonable rounding and for a value being expressed as a percentage instead
of a raw fraction:

```python
_NUMBER_PATTERN = re.compile(r"-?\d+\.\d+|-?\d+")

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_PATTERN.findall(text)]

def _value_matches(claimed: float, evidence_value: float) -> bool:
    candidates = {claimed, claimed / 100}
    for candidate in candidates:
        for decimals in range(0, 5):
            if abs(candidate - round(float(evidence_value), decimals)) < 1e-9:
                return True
    return False

def _ungrounded_claims(narrative: str, evidence: dict) -> list[float]:
    evidence_values = list(evidence.values())
    return [
        claim for claim in _extract_numbers(narrative)
        if not any(_value_matches(claim, v) for v in evidence_values)
    ]
```

This is not a check that trusts the model's word — it's a mechanical,
independent audit of the model's actual output text, run automatically
after every real call. The measured result, over 30 real clusters
deliberately selected to span the whole risk range (not just the
easiest-to-explain, highest-risk ones — `results/investigator_eval.md`):
**176 total numeric claims extracted across all 30 real LLM-generated
narratives, 0 of them ungrounded — a 100.00% groundedness rate.** The
report is careful to frame what this proves and doesn't: *"This is the
number that actually measures claude-sonnet-4-6's behavior under the
prompt's hard rule... a rate below 100% is a finding about the model's
behavior, not something to fix by loosening the claim extractor"* and,
separately, in the file's own closing limitation: *"The groundedness result
is one run of 30 clusters... that's one clean run, not a permanent
guarantee. The check should keep running on every future re-run, not be
treated as settled."* What this proves is narrow and specific: on this one
real run, against this one real model, under this specific hard-coded
system prompt rule, every single generated number was traceable to real
input data. It does not prove the model can never hallucinate a number
under different conditions — which is exactly why the check is built to
run on every future pipeline execution, not treated as a one-time
certification.

## SHAP explained from zero

**SHAP** stands for **SH**apley **A**dditive ex**P**lanations — a method
for answering the question "for this *one specific* prediction, how much
did each individual feature push the score up or down?" The idea comes
from a much older concept in game theory called a **Shapley value**:
imagine a group of players cooperating to earn some total payoff, and you
want to fairly divide credit for that payoff among the players based on how
much each one actually contributed. The Shapley value answers this by
considering every possible order in which players could join the group,
measuring how much the payoff changes each time a given player joins
(compared to the group without them), and averaging that marginal
contribution across all possible join-orders. Applied to a trained model:
the "players" are the model's input features, the "payoff" is the model's
prediction for one specific row, and a feature's **SHAP value** is (a
tractable approximation of) its fair share of credit for why the model's
prediction ended up wherever it did, relative to what the model would have
predicted with no information about that row at all.

This project uses SHAP's `TreeExplainer`, which is a fast, exact
algorithm specifically for tree-based models like the LightGBM boosters
this project trains — built once per process and cached, per
`dashboard_attribution.py`:

```python
@st.cache_resource(show_spinner="Building the SHAP explainer against the cluster model (once)...")
def get_shap_explainer(_cluster_model):
    import shap
    return shap.TreeExplainer(_cluster_model)
```

Every SHAP value here is expressed in **log-odds** (also called **margin**)
space — the same pre-sigmoid space introduced in Part 1.3, where a positive
value pushes toward fraud, a negative value pulls away from it, and the
values are additive: they sum up cleanly, rather than combining in some
more complicated nonlinear way. The **expected value** is the model's
"starting point" prediction with zero feature information at all — the
average prediction across the whole training set, in log-odds space. The
fundamental identity SHAP guarantees is:

```
sum(all SHAP values for this row) + expected_value == the model's raw log-odds prediction for this row
```

This project verified this identity is genuinely true against its own real,
trained model — not just assumed from the SHAP library's documentation.
Take a real example directly from this project's own dashboard: cluster
`74986`'s driving test-period transaction scored **0.3785** (a probability,
after the sigmoid). The dashboard's own SHAP breakdown for that exact
transaction reported: `expected_value = -10.4304`, cluster-derived
features summing to `+8.7792` log-odds, and transaction-level features
summing to `+1.1554` log-odds. Add those three numbers:

```
-10.4304 + 8.7792 + 1.1554 = -0.4958   (the total predicted log-odds/margin)
```

Now convert that back to a 0-to-1 score using the sigmoid function from
Part 1.3, `1 / (1 + e^(-x))`:

```
1 / (1 + e^(0.4958)) = 0.378528...
```

That rounds to **0.3785** — matching the model's real, independently-
computed score for this exact transaction, exactly. This isn't a coincidence
or an approximation that happens to look close: it's the SHAP identity
holding true, verified against this project's own real model and a real
transaction, not a toy example. The actual code that computes and verifies
this decomposition:

```python
def compute_shap_row(explainer, x_row: pd.DataFrame) -> tuple[pd.Series, float]:
    raw = explainer.shap_values(x_row)
    values = raw[1] if isinstance(raw, list) else raw
    values = np.asarray(values).reshape(-1)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    return pd.Series(values, index=x_row.columns), float(expected_value)
```

Finally, this project uses the same SHAP row to answer its own central
empirical question at the level of one individual decision, not just in
aggregate: *how much of this specific transaction's score came from
cluster-level signal vs. its own raw transaction-level features?*

```python
def txn_vs_cluster_split(shap_row: pd.Series, cluster_feature_columns: set) -> dict:
    cluster_cols = [c for c in shap_row.index if c in cluster_feature_columns]
    txn_cols = [c for c in shap_row.index if c not in cluster_feature_columns]

    cluster_sum = float(shap_row[cluster_cols].sum())
    txn_sum = float(shap_row[txn_cols].sum())
    cluster_abs = float(shap_row[cluster_cols].abs().sum())
    txn_abs = float(shap_row[txn_cols].abs().sum())
    total_abs = cluster_abs + txn_abs
    ...
```

For the same cluster-74986 example above, this split reported: cluster
features (just 10 columns) contributed **73%** of the total attribution
magnitude, while all 432 transaction-level features combined contributed
the remaining **27%** — a single number, computed fresh for every
individual decision shown on the dashboard, that makes this project's
central claim (cluster-derived signal matters, beyond what the raw
transaction alone tells you) visible and verifiable *per decision*, not
just as an aggregate PR-AUC lift buried in a results file.

---

# PART 9 — CALIBRATION, THRESHOLDS AND COST

## What calibration is, and what the Brier score measures

Part 1.3 introduced the core idea: a model's 0-to-1 output is only a real
**probability** if it's **calibrated** — if, among every transaction the
model scores around 0.30, roughly 30% of them are actually fraud. A model
can be extremely good at *ranking* (putting the riskiest transactions near
the top) while being badly wrong about what its raw numbers imply in an
absolute sense — ranking quality and calibration are genuinely different
properties, and this project measured both separately rather than assuming
good PR-AUC implies good calibration.

The **Brier score** is a standard way to summarize calibration+ranking
quality together into one number: it's the mean squared difference between
the predicted score and the actual outcome (0 or 1), averaged over every
test-set row — literally `mean((prediction - actual)^2)`. Lower is better;
0 is a hypothetical perfect model, and a model that always predicts the
test set's own base rate scores exactly `base_rate * (1 - base_rate)` as a
reference point (a "how would a naive constant-prediction model do"
baseline). This project's actual measured numbers, from
`run_pipeline.write_calibration` and `results/ablation.md`: the cluster
model's Brier score is **0.0200**, against a constant-prediction baseline
of **0.0332** (computed from the real test-set base rate, 0.0344) — the
real model is meaningfully better than guessing the base rate for every
row.

But a single overall Brier score, computed by averaging over *every*
prediction, can hide exactly the failure that matters most in this
project. Here's why, in the actual code that reveals it:

```python
def write_calibration(pipeline_data: PipelineData, trained: TrainedModels) -> None:
    n_bins = 15
    y_test = trained.y_test.to_numpy()
    y_score = trained.cluster_model.predict(trained.X_test_cluster)

    brier = float(brier_score_loss(y_test, y_score))
    ...
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_score, n_bins=n_bins, strategy="quantile"
    )
```

`calibration_curve` groups every test-set prediction into 15
**quantile bins** (bins with roughly equal *counts* of transactions in
each, not equal-width score ranges — necessary here because, at a ~3.5%
base rate, almost every score is close to 0, so equal-width bins would
dump nearly the whole test set into one bin and leave the higher-score
bins nearly empty). Within each bin, it computes the mean predicted score
and the actual observed fraction of fraud. If the model is well
calibrated, these two numbers should be close in every bin.

They are close in most of the 15 bins here — because most bins sit at very
low predicted scores, and at a ~3.5% base rate, predicting "close to zero"
for a population that's mostly zero is an easy target to hit accurately.
That's exactly why the equally-weighted average across all 15 bins is
**misleading if reported alone**, as `results/ablation.md` states directly:
*"Most of the 15 bins sit at very low predicted scores, where a
~3.5%-base-rate model is naturally easy to calibrate... so they pull the
average down."*

## What overconfidence of 0.13 in the threshold region means operationally

The bin that actually matters is the **highest-score bin** — the one
closest to where `policy.py`'s real thresholds operate. Its real, measured
numbers: mean predicted score **0.4795**, but the actual observed fraction
of positives in that bin is only **0.3509** — a gap of **−0.1286**, i.e.
the model is **overconfident by about 0.13** exactly in the region where
real decisions get made. In plain operational terms: when this model's
score says "I'm about 48% confident this is fraud," reality says the true
rate among transactions scoring that high is closer to 35%. The model
isn't lying about *which* transactions are riskier — its ranking is still
useful, which is what PR-AUC measures — but its raw number systematically
overstates *how* risky, specifically in the score range that actually
triggers `step_up` and `review` actions.

This has a direct, practical consequence for how `policy.py`'s thresholds
should be read and communicated. `REVIEW_THRESHOLD = 0.1843` should never
be presented to a business stakeholder, a regulator, or a customer-facing
explanation as "we estimate an 18.43% probability of abuse" — that
framing borrows the language of a calibrated probability the model does not
actually have. The correct framing, and the one `results/ablation.md`
insists on: *"policy.py's REVIEW_THRESHOLD (0.1843) should be read as an
arbitrary cut on this model's score scale, not as 'we estimate >18.43%
abuse risk' — scores in this upper range systematically overstate the true
positive rate."* The fix, if a genuinely calibrated probability were
needed for some other purpose (say, reporting an expected-loss estimate to
finance), would be **isotonic regression** or **Platt scaling** — both are
standard post-processing techniques that learn a separate mapping from raw
model score to calibrated probability, fit on a held-out slice of data —
and this project explicitly did *not* implement either, stating why
directly: *"model.py is frozen this session, and refitting a calibration
map changes how scores are produced, which is not something to add quietly
under a task that explicitly said do not retrain the model."* This is a
correct, disciplined refusal to quietly patch a known problem outside the
task's actual scope, not an oversight.

## How the thresholds were derived from the cost curve

`policy.py`'s two thresholds — `STEP_UP_THRESHOLD = 0.0103` and
`REVIEW_THRESHOLD = 0.1843` — are not hand-picked round numbers. They come
from a real script, `scripts/derive_policy_thresholds.py`, that sweeps a
fine grid of possible thresholds and finds, for each one, the
**cost-minimizing** point using `evaluate.cost_per_10k` (Part 6):

```python
THRESHOLDS = np.concatenate(
    [np.linspace(0.0, 0.05, 200), np.linspace(0.05, 1.0, 100)[1:]]
)

def _cost_minimizing_threshold(y_true, y_score, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    costs = [
        evaluate.cost_per_10k(y_true, y_score, t, cost_fn, cost_fp) for t in THRESHOLDS
    ]
    best_idx = int(np.argmin(costs))
    return float(THRESHOLDS[best_idx]), float(costs[best_idx])
```

Notice the grid itself: 200 points densely packed between 0.0 and 0.05,
and only 99 more spread across the much wider 0.05-to-1.0 range. This
non-uniform grid is deliberate — it exists because, at this dataset's ~3.5%
base rate, the cost-minimizing threshold turns out to live in that first,
densely-sampled low range (both real thresholds, 0.0103 and 0.1843, do),
so a plain evenly-spaced grid would waste most of its resolution on a part
of the score range where nothing interesting happens.

Both thresholds use the same `cost_fn=500` (the assumed cost of a missed
fraud case is identical regardless of which action would have caught it —
a missed case is a missed case) but **different** `cost_fp`:

- `STEP_UP_THRESHOLD` uses `cost_fp=5` — the exact same cost pair
  (`cost_fn=500, cost_fp=5`) already used to produce this project's
  headline ablation table (Part 6), i.e. `step_up` is modeled as a cheap,
  light friction event (an automated challenge).
- `REVIEW_THRESHOLD` uses `cost_fp=50` — ten times costlier, modeling a
  full manual review as a much more expensive false positive (real analyst
  time, a held transaction, a materially worse customer experience than an
  automated step-up challenge).

Both the 100:1 (`cost_fn:cost_fp` for step_up) and the extra 10x multiplier
for review are stated plainly, in `policy.py`'s own docstring, as
**illustrative** choices: *"The 10x multiplier is illustrative, not a
Razorpay figure — same caveat as every other cost assumption in this
project."* Changing either assumed cost would shift both thresholds, and
therefore every downstream `allow`/`step_up`/`review` decision this system
makes — which is precisely why this project treats them as parameters with
documented, honestly-labeled provenance, not as hidden magic numbers.

---

# PART 10 — QUESTIONS YOU MUST ANSWER COLD

Each answer below is short enough to say out loud in an interview without
sounding like you're reciting a document, but every number and claim is
real and traceable to a specific file.

**1. What does this project actually do?**
It resolves anonymous IEEE-CIS payment transactions into persistent
synthetic identities (`card1_addr1_origin_day`), links those identities
into a graph when they share a strong identifier (device, address+email,
or a card/bank/address combination), computes cluster-level features from
that graph causally (using only pre-cutoff data), and measures whether
adding those cluster features to a fraud classifier improves it over
transaction features alone — while keeping a deterministic policy layer
and an LLM explanation layer strictly separate from each other.

**2. What's the headline result?**
PR-AUC goes from 0.5646 (transaction features only) to 0.6322 (with
cluster features) on a single 80%-through-time temporal split — a
**+0.0676** lift. But about 84% of that lift comes from one feature,
`cluster_prior_fraud_share`, and the residual lift from the *other* nine
cluster features doesn't hold up across four rolling splits (mean
**+0.0060**, sign flips between splits) — so the honest headline is "one
specific, backward-looking feature helps reliably; the rest of the
structural graph signal doesn't, at this sample size."

**3. Why is `cluster_prior_fraud_share` not leakage?**
Because "leakage" specifically means information from *after* the cutoff
being scored leaking backward. `cluster_prior_fraud_share` is computed
inside `compute_cluster_features`, which filters every input row to
`TransactionDT < as_of` *before* computing anything — mechanically, not by
convention (Part 5). This was checked two ways: a unit test that plants a
huge, distinctively-valued future transaction and proves it changes
nothing, and a full sweep of all 155,579 real clusters comparing the
pipeline's reported value against an independently-recomputed pre-cutoff
value, bypassing `graph.py` entirely — zero mismatches. It's *not*
leakage. What it *is*, instead, is a legitimate feature whose predictive
power is partly circular: this dataset's labels themselves propagate
backward across a card once one chargeback is reported, so "has this
card/cluster already been caught before" is close to directly re-reading
the label-generation mechanism, not discovering new abuse. Leakage and
circularity are different problems — this project ruled out the first and
found, then reported honestly, the second.

**4. Why keep an entity definition (`card1_addr1_origin_day`) that's
known to over-merge different people?**
Because the two properties that matter for this project's actual use
case are *label consistency* (do a uid's transactions agree on outcome —
measured at 97.61% weighted) and *whether over-merging is even harmful
here* — and for a coordinated-abuse detector specifically, over-merging
distinct emails sharing one card/address/origin-day combination isn't
obviously wrong; it's arguably exactly the kind of signal worth
surfacing (a shared household, or coordinated card-testing). This project
never claims the uid identifies one verified physical person — it states
explicitly, everywhere the claim could be misread, that "cluster" here
means "transactions that plausibly share an underlying actor and are
internally label-consistent," not "one confirmed individual." Stability
without correctness, stated as a limitation rather than hidden.

**5. Why is the residual lift (without `cluster_prior_fraud_share`)
unstable?**
Because at this dataset's sample size, the effect being measured is close
to noise-level: mean +0.0060 across 4 splits with a spread of 0.0230 —
nearly four times the mean — and the *sign* itself flips (negative at 60%
and 90%, positive at 70% and 80%). A sign that depends on which slice of
the calendar you happen to test on is close to the textbook signature of
"this effect is not reliably distinguishable from zero at this sample
size," not "this effect is real but small." Adding richer topology
features (k-core depth, star ratio) didn't fix it either — same sign
pattern, slightly smaller mean (+0.0019).

**6. Why does a 3-uid cluster get `review` while a 50-plus-uid cluster gets
`allow`?**
Because `policy.decide` compares one number — the model's score for that
cluster's highest-scored transaction — against a fixed threshold; cluster
*size* barely factors into that score at all (its correlation with
`isFraud` is only 0.0309, versus `cluster_prior_fraud_share`'s 0.7797 —
`results/ablation.md`'s correlation table). And `cluster_prior_fraud_share`
is a *mean* across a cluster's members, not a count or a sum: a tiny
3-member cluster where 1 of 3 members has a prior fraud flag averages to
0.33 — a huge contribution. A 50-member cluster where zero members have
ever been flagged averages to 0.0 — no matter how many members it has.
Cluster size alone is nearly irrelevant to the model's score; what matters
is what *share* of a cluster's members carry fraud history, which a small,
concentrated cluster can hit far more easily than a large, mostly-clean
one.

**7. Walk me through the SHAP identity on a real example.**
For cluster 74986's driving transaction: `expected_value = -10.4304`
(log-odds), cluster features contribute `+8.7792`, transaction features
contribute `+1.1554`. Sum: `-10.4304 + 8.7792 + 1.1554 = -0.4958`. Push
that through the sigmoid, `1/(1+e^0.4958) ≈ 0.3785` — exactly matching the
model's real predicted score for that transaction. That's SHAP's
additivity guarantee, verified against this project's own real trained
model, not assumed from the library's documentation.

**8. Why PR-AUC and not ROC-AUC as the headline metric?**
Because at a ~3.5% base rate, the negative class is enormous (96.5% of the
data), and ROC-AUC's false-positive-rate axis is computed against that huge
pool — a model can look deceptively strong on ROC-AUC while still
generating an operationally unmanageable number of false alarms in
absolute terms. PR-AUC's precision axis is sensitive to exactly that
failure mode, which is what an analyst team actually experiences.

**9. What would a random model score on PR-AUC, and why not 0.5?**
Approximately the base rate — about 0.035 here, not 0.5. This project's
own test proves it: `test_pr_auc_uncorrelated_scores_are_near_base_rate`
generates uncorrelated random scores against a 3.5%-base-rate label and
gets ~0.035 back. ROC-AUC is the metric where 0.5 means random; PR-AUC's
random floor is the base rate.

**10. Why can't the split be random?**
Because the model's features aggregate a cluster's transactions together,
and clusters span time — a random shuffle would scatter one cluster's
earlier and later transactions across both train and test, letting the
model implicitly see information from a cluster's *future* transactions
while training, then get evaluated on that same cluster's *earlier* ones.
That's temporal leakage baked into the split itself. `CLAUDE.md`'s rule is
categorical: temporal splits only, on `TransactionDT`.

**11. What does `as_of` actually do, mechanically?**
It's the earliest `TransactionDT` in the test set. `compute_cluster_features`
filters its input to `TransactionDT < as_of` as its very first operation,
before computing any aggregate — so every downstream feature, whether
being computed for a train-period or test-period transaction, only ever
sees data strictly earlier than the moment the test period begins.

**12. How do you know the graph itself doesn't leak test-period structure?**
`build_entity_graph` is called on `train_df` only in `load_and_prepare` —
`test_df` is never passed to it, so no test-period transaction can
contribute a node or an edge, structurally, not by a runtime check. This
was verified concretely too: 0 nodes in the real graph correspond to a
uid with zero train-period transactions.

**13. What's `max_degree` and why is it 20, not the function's own default
of 1000?**
It's the hub guard: any value shared by more than `max_degree` uids is
excluded from linkage as "too common to be evidence of a relationship."
Sweeping this on the real data found a sharp phase transition: at 1000
(the function's own default), 64% of all uids collapse into one giant
connected component; at 20, the largest component is just 0.06% of all
uids. This project uses 20, deliberately conservative, because a single
supercluster destroys every graph-derived feature's usefulness far more
than missing a few genuinely large rings would.

**14. What's the difference between `star_ratio` and `cluster_edge_density`,
and why do you need both?**
`star_ratio` (highest single member's degree, divided by cluster size) and
a mutually-connected clique of the same size produce the *identical*
`star_ratio` by construction — this project's own test proves it (a 4-node
star and a 4-node clique both score 0.75). What tells them apart is edge
density (0.5 for the star, 1.0 for the clique) and k-core number (1 for the
star — trees have no cycle to sustain a deeper core — vs. 3 for the
clique). Reading `star_ratio` alone would conflate a hub-and-spoke shape
with a tight mutual ring.

**15. Neither topology feature helped PR-AUC. Was that wasted work?**
No — it's a negative result, reported plainly, and negative results that
close off a plausible hypothesis are real findings, not wasted effort. The
prior finding (aggregate cluster-structure features don't show a stable
lift) left open the question "maybe the aggregates were just too crude —
richer shape features might do better." Testing that directly, and finding
the same unstable, sign-flipping pattern even with topology added, closes
that specific door rather than leaving it as an untested "maybe."

**16. What's the Brier score, and is 0.0200 good?**
It's the mean squared error between predicted score and actual outcome (0
or 1) — lower is better. 0.0200 beats the constant-prediction baseline of
0.0332 (always guessing the 3.44% test-set base rate), so overall the model
is better than a naive constant. But the *overall* Brier score is
misleading here specifically, because most of the test set sits at very
low, easy-to-calibrate scores and pulls the average down — what actually
matters is calibration in the high-score region, which is worse (next
question).

**17. Is the model's score a real probability?**
No, not in the region that matters. The highest-score bin predicts a mean
score of ~0.48, but the real observed fraud rate in that bin is only
~0.35 — the model is overconfident by about 0.13 exactly where
`policy.py`'s thresholds operate. `REVIEW_THRESHOLD=0.1843` should be read
as an arbitrary cut on this model's own score scale, not as "an 18.43%
probability estimate."

**18. How were `STEP_UP_THRESHOLD` and `REVIEW_THRESHOLD` chosen?**
By sweeping a fine grid of candidate thresholds and finding the
cost-minimizing point via `evaluate.cost_per_10k`, using `cost_fn=500`
for both (a missed fraud case costs the same either way) but different
`cost_fp` — 5 for step_up (a cheap automated challenge) and 50 for review
(a 10x-costlier full manual review). Both cost figures are explicitly
labeled illustrative, not real Razorpay numbers.

**19. What's `cost_per_10k`, and why does the "optimal" threshold flag ~37%
of legitimate transactions?**
It's `(FN * cost_fn + FP * cost_fp) / n * 10000` — the expected weighted
cost of the model's mistakes, normalized per 10,000 transactions. The
100:1 assumed cost ratio (missing fraud costs 100x a false alarm) means the
cost-minimizing threshold sits very low on the score scale, which
mathematically forces a high false-positive rate as the price of catching
almost all fraud (95.3% recall at that point) — that's a direct, correct
consequence of the assumed cost ratio, not a bug; a less aggressive ratio
would move the threshold and shrink that false-positive rate substantially.

**20. Why keep `policy.py` and `investigator.py` completely separate?**
Because an LLM's output has no determinism guarantee and no auditable,
mechanically-reproducible reasoning chain — using it to influence which
action gets taken on a real customer's transaction would make outcomes
depend on generated-text variance, which is unacceptable for a compliance-
sensitive financial decision. `policy.py`'s decision is pure score-vs-
threshold arithmetic, reproducible byte-for-byte forever. This is enforced
two ways: an AST-level static check that `policy.py`'s source contains no
import of `investigator`, and a behavioral test proving decisions are
identical whether or not the investigator module (or an API key) is even
available.

**21. How do you know the LLM narratives aren't hallucinating numbers?**
Every generated narrative is checked programmatically: every number
extracted from the text (via regex) must match some value in the evidence
JSON the model was actually given (allowing reasonable rounding or a
percentage form). Measured over 30 real clusters spanning the full risk
range: 176 total numeric claims, 0 ungrounded — 100% groundedness. But
this is one clean run, not a permanent guarantee, and the check is
designed to keep running on every future pipeline execution rather than be
treated as settled.

**22. This project once had a groundedness incident — what happened?**
An identity-linked (workspace-scoped) Anthropic API key requires an extra
`anthropic-workspace-id` header on every request; without it, every single
call failed silently, and the investigator layer's graceful-fallback
design (never crash the pipeline for a bad key) meant this failure was
invisible — a report was produced claiming "at least one explanation used
the real LLM path" while all 30 of 30 had actually fallen back to the
deterministic template. The fix wasn't just adding the header — it was
making fallback caused by a *real failure* observably different from
fallback caused by *no key at all*: every fallback now logs the specific
exception to stderr and records it on the result object (`.error`), so a
report built from real outcomes can never again claim success while
everything actually failed.

**23. What does the queue-level evaluation (`results/queue_eval.md`) show,
and why does it matter that the priority score didn't win?**
It measures whether an analyst working the top-K flagged *clusters* would
actually find real abuse near the top — a more operationally realistic
question than transaction-level PR-AUC. Base rate: only 9 of 494
qualifying clusters contain fraud (1.8%). Both the system's hand-weighted
priority ranking and a naive mean-transaction-score baseline land far
above that base rate (3.3x-22x lift depending on K) — but the priority
ranking is behind or tied at all 4 K values tested against the naive
baseline. With only 9 positive clusters total, that gap is within the
noise this sample size can produce, and the report says so plainly rather
than either declaring victory or panicking — it's a genuine null finding,
reported honestly, not hidden because it's unflattering.

**24. You tried five different priority-ranking formulas afterward — did
any of them clearly win?**
No, and that itself is the finding: the best-performing variant differs by
K (a "max transaction score in cluster" ranking led at K=10 and K=50, "mean
transaction score" led at K=25, the original priority score led at K=100),
and the spread between the best and worst variant's cluster-count at any
given K is only 1-4 clusters — the same order of magnitude as the noise
already flagged in `results/queue_eval.md`. The honest conclusion is "not
enough positive examples in this test window to distinguish these five
rankings with any confidence," not "here's a better formula" — and the
report explicitly declines to adopt any of them as a replacement.

**25. What's the single most important thing you'd want an interviewer to
take away from the stability finding?**
That a good-looking single-split number was deliberately re-tested under
harder conditions (four splits, then topology features on top), found not
to hold up for the residual (non-dominant-feature) portion of the claim,
and the project's own documentation was rewritten to say so — more
strongly than the original claim, not softened. That discipline — being
harder on your own favorable result than on an unfavorable one — is the
actual skill being demonstrated, more than any single metric in the table.

**26. What would you do differently, or next, if you kept working on
this?**
Three concrete things: (1) get access to real ring-labeled ground truth or
partner with an actual investigations team to validate cluster-level
findings against confirmed outcomes — this project has zero ring-level
ground truth anywhere, only transaction-level chargeback labels, which is
explicitly called out as a limitation; (2) implement a proper held-out
isotonic or Platt calibration layer, kept structurally separate from the
frozen scoring model, so operational thresholds could be communicated as
real probabilities instead of arbitrary score cuts; (3) get a longer
observation window or more test-period volume specifically to fix the
9-positive-cluster small-sample problem in the queue-level evaluation,
which currently can't distinguish between competing ranking strategies
with any confidence.

**27. How would this scale to roughly a billion transactions a quarter?**
This project's own real benchmark: building the entity graph and
computing cluster features for 472,432 train-period transactions took
6.64s and 19.71s respectively — fast at this scale, but ~1B/quarter is
roughly 2,117x this benchmark's train set, and the real cost is worse than
linear, not just larger: `max_degree`'s effect on cluster size is a phase
transition (Part 4), and at greater scale, more identifier values would
cross whatever hub threshold is chosen, risking the same giant-component
collapse this project already found once, at a far more expensive scale to
detect and recover from. Three concrete changes this scale would require,
none implemented here: incremental graph updates instead of full rebuilds
(the current `build_entity_graph` has no notion of "since last run"),
approximate/distributed connected-components computation instead of exact
single-machine `networkx` (the current implementation), and sharding the
in-memory graph and feature table across multiple machines, which raises a
real, unsolved design question here — how a device or address linking two
uids in *different* shards gets detected and reconciled.

**28. What's the actual latency profile, and why does it matter that batch
and inline are measured separately?**
Graph construction and cluster feature computation are batch — periodic
(e.g. nightly), against historical data, never on a live request's
critical path. Per-transaction scoring is inline — look up a uid's
precomputed features (a cache/feature-store read, not a recomputation) and
call the model, real measured p50 of 51.9ms, p95 of 58.1ms, p99 of 65.4ms
over 1,000 real single-row scoring calls. These two numbers answer
completely different capacity questions — batch bounds how often the graph
can be refreshed; inline bounds request latency — and reporting one
average across both would obscure that they're roughly three orders of
magnitude apart and governed by entirely different constraints.

**29. Why is `~11%` of the dataset getting no uid at all a real problem,
not just a footnote?**
Because that population isn't a random slice of traffic — it's the
*highest-risk* slice: 11.63% fraud rate among no-uid rows vs. 2.46% among
uid'd rows (`results/uid_validation.md`). Silently dropping unresolvable
rows, which many systems do by default, would systematically exclude the
most dangerous traffic from every cluster-based signal. This project keeps
those rows in training and evaluation with null cluster features rather
than dropping them — an explicit design decision, not an oversight, stated
directly in `run_pipeline.py`'s own code comments.

**30. If someone challenged you that this whole project is "just measuring
the labels' own generation process," how would you respond?**
I'd agree with the core of that critique for the *dominant* feature
specifically, and say so unprompted — this project's own Limitations
section states it in exactly those terms: `cluster_prior_fraud_share`
"measures label propagation across a card, not independently-discovered
abuse." What I'd push back on is treating that as an oversight rather than
a finding: this project traced that exact feature end to end, proved (not
assumed) it doesn't leak future information, quantified precisely how much
of the headline number it accounts for (84%), tested whether anything
*else* in the feature set carries independent, stable signal (twice —
aggregates, then topology), found it largely doesn't at this sample size,
and reported the corrected, less flattering conclusion in the same document
that reports the original number. The honest final claim isn't "we
discovered coordinated abuse" — it's "we built a defensible pipeline, and
what it mostly measures is whether a card has already been caught before;
independently discovering *new* abuse from graph structure alone remains
unproven at this sample size." That's a narrower claim than the headline
number alone would suggest, and stating it precisely, unprompted, is the
actual point of doing this kind of self-audit in the first place.

---

# PART 11 — GLOSSARY

**Ablation / ablation study** — comparing two systems that are identical
except for one deliberately-removed or deliberately-added component, so
any measured difference can be attributed to that one component alone. See
Part 6.

**Abstract syntax tree (AST)** — a structured, parsed representation of
source code that a program can inspect mechanically (e.g. "does this file
contain an import of module X"), rather than searching the raw text. Used
by `tests/test_policy.py` to prove `policy.py` never imports
`investigator.py`.

**Abuse score / model score** — the 0-to-1 number a trained model outputs
for a transaction; higher means "looks more like fraud." Not automatically
a probability (see Calibration).

**Accuracy** — the fraction of predictions a model got right. Useless
under class imbalance: a model that always predicts "not fraud" scores
96.5% accuracy on this dataset while catching zero fraud. See Part 1.4.

**addr1 / addr2** — coded address/region columns in the transaction file.
`addr1` is finer-grained (hundreds of distinct values); `addr2` is coarser
(near-constant, closer to a country-level code).

**Allow** (policy action) — `policy.py`'s decision when a score is below
`STEP_UP_THRESHOLD`: let the transaction through with no added friction.

**as_of** — the timestamp cutoff (the first moment of the test period)
passed into `compute_cluster_features` so every feature it computes only
ever uses transactions strictly before that moment. See Part 5.

**Audit record** — a flat, JSON-serializable dict (`policy.build_audit_record`)
capturing one decision's score, threshold, action, reason, and feature
values at decision time, for traceability.

**Average precision** — the exact metric scikit-learn's
`average_precision_score` computes; this project's `pr_auc` is a direct
wrapper around it. Equivalent in practice to PR-AUC.

**Bagging fraction / feature fraction** — LightGBM hyperparameters
controlling what fraction of rows (`bagging_fraction`) or columns
(`feature_fraction`) each individual tree is allowed to see; this project
uses 0.8 for both, so no single tree can over-rely on one feature or one
cluster of rows.

**Base rate** — the raw frequency of the positive class in a population
(here, ~3.5% dataset-wide, 3.44% on the test split, 1.8% among
queue-eval's qualifying clusters). Every precision-style metric in this
project is reported alongside its base rate, since a bare precision
number is meaningless without it.

**Baseline model** — the model trained on transaction features only, with
no cluster-derived features — the "before" side of this project's central
ablation.

**Boosting round** — one iteration of gradient boosting: build one small
tree, add its (scaled-down) prediction onto the running total. This
project uses 300 rounds (`NUM_BOOST_ROUND`).

**Brier score** — mean squared error between a model's predicted score and
the actual 0/1 outcome, averaged over a test set. Lower is better; measures
calibration and ranking together. See Part 9.

**C columns (C1-C14)** — 14 anonymized counting features in the
transaction file; the competition does not disclose exactly what each one
counts.

**Calibration** — whether a model's raw score matches real-world outcome
frequency (a score of 0.30 should mean "about 30% of these are actually
positive"). Distinct from ranking quality. See Parts 1.3 and 9.

**Calibration curve / reliability curve** — a plot (and the underlying
computation) of mean predicted score vs. observed fraction of positives,
within quantile bins, used to visualize and measure calibration.

**Card1-card6** — the six card-attribute columns in the transaction file.
`card4`/`card6` are human-readable (network/type); `card1`, `card2`,
`card3`, `card5` are anonymized numeric codes. `card3`/`card5` behave as
near-fixed sub-attributes of the card, confirmed empirically (they never
varied across the 20 largest resolved identities).

**Category dtype** — pandas' memory-efficient encoding for a text column
with a small, repeating set of distinct values (e.g. `ProductCD`). Used by
`src/data.py`'s downcasting step and by `src/model.py` when preparing
LightGBM's native categorical feature handling.

**Causal (feature computation)** — a feature computed using only
information that would have existed at the actual moment a real decision
had to be made — no information from the future relative to the row being
scored. See Part 5.

**Chargeback** — a bank/network-mediated reversal of a payment, typically
initiated by the cardholder disputing the charge. This dataset's
`isFraud=1` label is chargeback-derived and propagates backward across a
card's history once one transaction on it triggers a chargeback. See Part
2.

**Class imbalance** — when the two outcomes being predicted occur at very
different rates (here, ~3.5% fraud vs. ~96.5% legitimate). Makes accuracy
useless and motivates precision/recall/PR-AUC instead. See Part 1.4.

**Cluster** — in this project's terminology, one connected component of
the entity graph — a group of uids linked, directly or via intermediate
members, by at least one linkage rule. Not a claim about verified
real-world identity (see Over-merging).

**cluster_prior_fraud_share** — the single dominant engineered feature:
the share of a cluster's member uids that have ever had a fraud-labeled
transaction, as of the causal cutoff. Accounts for about 84% of this
project's headline PR-AUC lift, and is backward-looking/partly circular
with respect to how this dataset's labels are generated. See Parts 6 and
7.

**Collision check** — this project's test of whether a resolved identity
(uid) actually corresponds to one real person, by checking whether
secondary fields (especially email domain) vary within a single uid's
transactions. Found 10 of the 20 largest uids have more than one distinct
email domain — the basis of the over-merging finding. See Part 3.

**Connected component** — a maximal group of graph nodes reachable from
one another by following edges, with no edges reaching outside the group.
A node with no edges is its own (singleton) component. See Part 4.

**Cost_fn / cost_fp** — the assumed dollar-equivalent cost of a false
negative (a missed fraud case) and a false positive (a wrongly-flagged
legitimate transaction), respectively. Always parameters, never hardcoded,
in `evaluate.cost_per_10k`. Every specific value used in this project
(500/5, 500/50, and so on) is explicitly labeled illustrative, not a real
Razorpay figure.

**Cost per 10k (txns)** — `(FN * cost_fn + FP * cost_fp) / n * 10000`, this
project's business-facing summary metric, normalized to a fixed
transaction-volume basis. See Part 6.

**D columns (D1-D15)** — 15 anonymized time-delta ("days since some event
X") features in the transaction file. `D1` is investigated at length in
this project (Part 3) and behaves as "days since this card was first
seen."

**Decision tree** — the simplest model of the kind used here: a sequence of
yes/no questions about the input, ending in a prediction. See Part 1.2.

**Degree (of a graph node)** — how many edges touch that node.

**Density (of a graph/cluster)** — actual edges divided by the maximum
possible edges among that many nodes (`V*(V-1)/2`). 1.0 means every member
is directly linked to every other (a clique).

**DeviceInfo** — a free-text device-fingerprint string in the identity
file, present for ~24% of transactions; the basis of this project's
`device_info` linkage rule.

**Downcast** — reducing a column's numeric precision (`float64` →
`float32`) or converting a low-cardinality text column to `category`, to
shrink memory usage without materially changing the data's meaning. See
`src/data.py`, Part 2.

**Edge** — a connection between exactly two nodes in a graph, here
representing "these two uids share a strong identifying signal." See Part
4.

**Entity graph** — the graph whose nodes are uids and whose edges come
from `LINKAGE_RULES`; built once from train-period data only
(`build_entity_graph`).

**Entity resolution** — the process of grouping raw transactions into
persistent synthetic identities (uids) when no real customer ID exists.
See Part 3.

**Evidence (investigator.py)** — the flat, JSON-serializable dict of real
feature values given to the LLM as its only source of numeric facts; every
number in a generated narrative must trace back to a value here.

**Expected value (SHAP)** — the model's average prediction (in log-odds
space) across the whole training set — the "starting point" before any of
a specific row's features are accounted for.

**False negative (FN)** — real fraud the model failed to flag. See Part
1.5.

**False positive (FP)** — a legitimate transaction the model wrongly
flagged. See Part 1.5.

**False positive rate (FPR)** — `FP / (FP + TN)`, measured over the
negative class only. The x-axis of an ROC curve; the metric
`recall_at_fpr` holds fixed while measuring recall.

**Feature** — one piece of information about a transaction, known before
the outcome, used as model input (e.g. `TransactionAmt`, `card1`,
`cluster_prior_fraud_share`).

**Feature fraction** — see Bagging fraction.

**Feature importance / gain** — how much a feature reduced the training
loss across every tree that used it, summed over all boosting rounds; this
project's way of ranking which features the trained model actually relied
on most.

**Gradient boosting** — building many small, weak trees in sequence, each
one correcting the specific errors the ensemble-so-far still makes, then
summing all their (scaled-down) contributions. See Part 1.2.

**Groundedness** — whether every number stated in an LLM-generated
narrative traces back to a real value in the evidence it was given.
Measured programmatically, not assumed. See Part 8.

**Hallucination** — an LLM stating something (here, specifically a number)
that isn't actually supported by its real input. This project's
groundedness check exists specifically to catch this failure mode.

**Hub / hub guard** — a value shared by an implausibly large number of
uids (a generic device string, a common email provider) that is excluded
from linkage entirely, since it's evidence of commonness, not of a specific
relationship. See Part 4.

**Hyperparameter** — a setting that controls *how* a model trains (e.g.
`learning_rate`, `num_leaves`), as opposed to a *parameter*, which the
training process itself learns from data.

**IEEE-CIS** — the real, published Kaggle fraud-detection competition
dataset this project uses (a collaboration between IEEE and the fraud
prevention company Vesta Corporation).

**Isolated node** — a graph node with degree 0 — linked to no one, its own
connected component.

**isFraud** — the label column: 1 if this transaction was (or was later
attributed to a card that was) chargeback-reported as fraud, 0 otherwise.

**K-core / core number** — the largest densely-connected sub-structure a
node belongs to, where every remaining node has at least `k` neighbors
within that sub-structure. A tree/star can never exceed core number 1 (no
cycle); a clique's every member has a core number equal to (size − 1). See
Part 4.

**Label** — the true, known answer for a training example (here,
`isFraud`) — not available at prediction time in a real deployment, only
during training/evaluation on historical data.

**Label purity** — for a resolved identity (uid), whether all of its
transactions share the same fraud label. Measured at 97.61% weighted
across multi-transaction uids in this dataset. See Part 3.

**Leakage** — information that would not actually be available at
prediction time sneaking into a feature or a split, inflating reported
metrics beyond what a real deployment could achieve. See Part 1.6.

**Learning rate** — how much each new tree's correction is scaled down
before being added to the running prediction total in gradient boosting;
smaller steps, more of them, tend to generalize better.

**Left join / inner join** — two ways of combining two tables on a shared
key. A left join keeps every row from the left table (filling in nulls
where the right table has no match); an inner join keeps only rows present
in both. `src/data.py` uses a left join specifically because an inner join
would silently discard the ~76% of transactions with no identity record.

**LightGBM** — the specific, fast gradient-boosting-over-decision-trees
library this project trains its models with.

**Lift (over base rate)** — a metric divided by the relevant base rate,
expressing "how many times better than random" a result is. Used
throughout `results/queue_eval.md` and `results/priority_variants.md`.

**Linkage rule** — one of three specific rules (`device_info`,
`addr1_email`, `card_bank_addr`) that link two uids into an edge when they
share the specified combination of fields. See Part 4.

**LLM (large language model)** — the AI model (here, `claude-sonnet-4-6`)
used by `investigator.py` to generate narrative explanations. Never used
to make a decision — see Policy/investigator separation.

**Log-odds / margin (space)** — the pre-sigmoid, unbounded real-number
space a model's raw prediction lives in before conversion to a 0-to-1
score; SHAP values are expressed in this space because they're additive
here. See Parts 1.3 and 8.

**M columns (M1-M9)** — 9 anonymized match-flag columns in the transaction
file (whether some pair of fields matched).

**Max_degree** — the hub-guard parameter: a value shared by more uids than
this is excluded from linkage. This project uses 20 (not the function's
own default of 1000) after finding a sharp phase transition in the real
data. See Part 4.

**Model** — mechanically, a fixed function (fixed input features, learned
internal parameters, a fixed prediction procedure) that maps input data to
an output number. See Part 1.1.

**Node / vertex** — one "thing" in a graph; here, one uid.

**Node degree** — see Degree.

**Over-merging** — when a synthetic identity key (uid) groups together
transactions from more than one real underlying entity. Confirmed in this
project via the collision check (10 of the 20 largest uids show more than
one email domain). Kept as a design choice, not fixed, since the merged
grouping is itself treated as a coordinated-abuse signal here. See Part 3.

**Overfitting** — a model learning noise/coincidence specific to its
training sample rather than the general underlying pattern, causing it to
perform worse on new, unseen data. See Part 1.6.

**P_emaildomain** — the purchaser's email domain column; part of the
`addr1_email` linkage rule.

**Parameter** — an internal number a model's training process chooses,
encoding "how much each feature matters and in what combination." Not set
by hand (contrast with Hyperparameter).

**Phase transition** — a sudden, large qualitative change in behavior from
a small change in an input parameter, rather than a smooth, gradual one.
This project found one in `max_degree`'s effect on the largest graph
component (a five-fold jump moving from 30 to 35).

**Platt scaling / isotonic regression** — standard post-processing
techniques that learn a separate mapping from a model's raw score to a
genuinely calibrated probability, fit on held-out data. Identified as the
correct fix for this project's calibration gap, but deliberately not
implemented (would require retraining/refitting, out of this project's
scope under its frozen-model rules). See Part 9.

**Policy / investigator separation** — the architectural rule that
`policy.py` (deterministic, decides) and `investigator.py` (LLM-based,
explains/prioritizes only) must never be merged, enforced both statically
(AST import check) and behaviorally (identical decisions with the LLM
layer disabled). See Part 8.

**PR-AUC** — area under the precision-recall curve; this project's
headline ranking-quality metric. 1.0 is perfect, ~base-rate is random
guessing. See Part 1.5.

**Precision** — of everything flagged, how much was actually positive:
`TP / (TP + FP)`. See Part 1.5.

**Priority score** — `investigator._priority_score`'s heuristic ranking
number for a cluster (`cluster_prior_fraud_share * 100 + burst_concentration
* 10 + min(txn_count, 100) * 0.1`), used to order an investigation queue.
Not a policy decision. Found not to beat a naive mean-score baseline at
queue-level ranking (`results/queue_eval.md`), though the sample size (9
positive clusters) can't confidently distinguish ranking strategies.

**ProductCD** — a coded product-category column (`W`, `C`, `R`, `H`, `S`)
in the transaction file.

**Quantile bins** — bins containing roughly equal *counts* of observations
(as opposed to equal-width score ranges), used for calibration curves
here because scores concentrate near zero at this base rate.

**Recall** — of everything actually positive, how much did the model
catch: `TP / (TP + FN)`. See Part 1.5.

**Review** (policy action) — `policy.py`'s decision when a score meets or
exceeds `REVIEW_THRESHOLD` (0.1843): route to full manual review.

**ROC-AUC** — area under the receiver-operating-characteristic curve
(recall vs. false-positive rate). Less informative than PR-AUC under
severe class imbalance. See Part 1.5.

**Seed** — a fixed starting value for a random-number generator, making
every "random" choice a training run makes fully reproducible.

**SHAP (SHapley Additive exPlanations)** — a method, based on Shapley
values from game theory, for attributing how much each individual feature
pushed one specific prediction up or down relative to the model's average
prediction. See Part 8.

**Shapley value** — a game-theory concept: a fair division of credit for a
group's total payoff among its members, based on each member's average
marginal contribution across every possible order in which they could have
joined.

**Sigmoid function** — `1 / (1 + e^(-x))`; converts an unbounded log-odds
number into a bounded 0-to-1 score, preserving order.

**Sign flip** — when a measured effect is positive on some data subsets
and negative on others, a strong signal the true effect may not be
reliably distinguishable from zero at the sample size tested. Found in
this project's residual (non-`cluster_prior_fraud_share`) cluster-feature
lift across rolling splits. See Part 7.

**Singleton** — a connected component containing exactly one node (an
isolated uid with no linkage edges).

**Star ratio** — the highest single member's graph degree in a cluster,
divided by the cluster's size; close to 1 for a hub-and-spoke shape, but
also close to 1 for a same-size clique — must be read alongside edge
density and k-core number to tell the two shapes apart. See Part 4.

**Step_up** (policy action) — `policy.py`'s decision when a score meets or
exceeds `STEP_UP_THRESHOLD` (0.0103) but is below `REVIEW_THRESHOLD`:
apply a lighter, automated additional-verification challenge.

**Temporal leakage** — leakage specifically caused by future-relative-to-
the-decision information entering a feature or a split, in a
time-ordered prediction task. See Part 5.

**Temporal split** — dividing data into train/test by time (everything
before a cutoff vs. everything after), never by random shuffling, so a
model is only ever evaluated on genuinely later data than it trained on.
See Parts 1.6 and 5.

**Threshold** — the score value at or above which a model's continuous
output gets converted into a discrete decision (flag/don't flag, or which
of several actions to take).

**Topology (graph)** — the shape of a graph's connections (hub-and-spoke
vs. clique vs. chain), as distinct from simple aggregate measures like
size or density. This project's `k_core_number` and `star_ratio` features
attempt to capture it. See Parts 4 and 7.

**TransactionDT** — the dataset's raw timestamp column (seconds since an
arbitrary fixed reference point, not a real calendar date) — the field
every temporal split and every causal filter in this project is anchored
to.

**True negative (TN)** — a legitimate transaction correctly left
unflagged.

**True positive (TP)** — real fraud correctly flagged.

**uid** — this project's synthetic, persistent identity key,
`card1_addr1_origin_day`, used in place of a real (nonexistent) customer
ID. See Part 3.

**V columns (V1-V339)** — 339 fully anonymized, proprietary engineered
features contributed by Vesta Corporation; the largest single column
family in the dataset, with opaque but real predictive signal (`V258`
ranks as the second-most-important feature in this project's trained
cluster model).

**Vesta Corporation** — the fraud-prevention company that contributed the
underlying transaction data to the IEEE-CIS competition.

**Workload / efficiency (queue-eval terms)** — `workload` is how many
test-period transactions an analyst would have to review to work through
the top-K flagged clusters; `efficiency` is fraud transactions surfaced
divided by that workload — how much of the review effort was actually
worthwhile. Both reported in `results/queue_eval.md` and
`results/priority_variants.md` alongside precision, since precision alone
doesn't capture how much total review effort a ranking demands.

---

*Every number in this document traces to a file in `results/`, `src/`,
`tests/`, `app.py`, `dashboard_attribution.py`, `ARCHITECTURE.md`,
`README.md`, or a direct query against the real `data/train_transaction.csv`
run while writing this guide. Nothing here describes code that wasn't
opened and read. No pipeline file was modified to produce this document.*
