# Dev log

Honest, specific notes on what was built, what was surprising, and what broke
or needed correcting. Written as work happens, not cleaned up afterward.

## 2026-08-31 — Session setup (branch, CLAUDE.md rules)

Branched to `uid-validation` off `main`. Added two rules to CLAUDE.md:
test_transaction.csv has no labels (it's the Kaggle competition's actual
holdout, never released to competitors), so every train/test split in this
project has to happen inside train_transaction.csv; and synthetic label/score
arrays used as metric-function test fixtures (e.g. `np.array([0,1,1,0])`) are
explicitly not "fraud generation" under the defense-only rule. `.venv/` was
already covered by the existing `.gitignore`, so nothing to add there.

Worth recording from the *previous* scaffolding session even though it
predates this devlog: the original `.gitignore` had a bare `data/` rule plus
`!data/.gitkeep`, intending to keep the placeholder trackable while ignoring
real data. That doesn't work — git's negation rule cannot re-include a file
inside a directory that's already excluded at the directory level, so
`data/.gitkeep` was silently still ignored. Fixed by changing the pattern to
`data/**` (ignores contents, not the directory entry itself), which lets the
negation actually take effect. Verified with `git check-ignore -v` and
`git add --dry-run`.

## 2026-08-31 — Task 1: src/data.py

Implemented `load_transactions`: left-join train_transaction.csv with
train_identity.csv on TransactionID (left, not inner, since identity covers
only ~24% of transactions), downcast float64->float32 and low-cardinality
(<=50 unique values) object columns to category, and raise a
`FileNotFoundError` naming `scripts/download_data.sh` when either CSV is
missing.

Added a root-level `conftest.py` (empty file) so `tests/` can `import src.*`
regardless of how pytest is invoked. Without it, pytest's default "prepend"
import mode only adds the `tests/` directory itself to `sys.path` (since it
has no `__init__.py`), not the repo root — `import src.data` would fail
depending on cwd. This wasn't something the task called out explicitly; found
it by reasoning through pytest's import mechanics rather than by hitting the
failure, so it's untested against every possible invocation style, just the
common ones (`pytest`, `python -m pytest`, both from repo root).

The category-cutoff constant (50 unique values) is a judgment call, not
something tuned against real data — no CSVs are present in this environment
yet (see Task 3 below), so the actual cardinality of columns like ProductCD,
card4/card6, M1-M9, or the id_* columns hasn't been checked. It may need
revisiting once real data is loaded; DeviceInfo in particular is known from
the competition's public documentation to have far more than 50 distinct
values and will correctly stay `object` under this rule.

Tests cover only the missing-file error path (3 cases: both missing, identity
missing, transaction missing), per the task's explicit scope — the
load/join/downcast happy path needs the real dataset.

## 2026-08-31 — Task 2: evaluate.py implemented and frozen

This is the metrics module that can never be edited later to look better, so
getting the design right now mattered more than usual. Notes on the real
decisions made, since the task description left some of them open:

- **temporal_train_test_split**: a naive "take the row at the (1-test_size)
  quantile" split can land in the middle of a group of rows sharing one
  TransactionDT value, putting identical timestamps on both sides of the
  boundary. Implemented as a backward walk over *unique* TransactionDT values
  from most recent to oldest, accumulating each group's full row count atomically,
  stopping once the accumulated count reaches the target test size. This
  guarantees no timestamp group is ever split, at the cost of the realized
  test fraction sometimes drifting from the requested `test_size` when a
  large group straddles the boundary (tested explicitly with a synthetic
  20-row tied group). Added an internal `assert train[dt_col].max() <
  test[dt_col].min()` inside the function itself as a cheap self-check, given
  how much rides on this invariant actually holding.
- **evaluate_model's signature** needed `cost_fn`/`cost_fp` parameters that
  weren't in the task's literal function-reference list, since it has to call
  `cost_per_10k` internally and the instructions were explicit that costs must
  never be hardcoded. Added them as optional kwargs defaulting to a neutral
  1:1 ratio (not a guessed rupee amount) so callers can override with real
  business costs later.
- **evaluate_model calls `model.predict(X_test)` directly** (LightGBM's native
  Booster API), not `src.model.predict()`. src/model.py is explicitly out of
  scope for this session and still raises NotImplementedError, so routing
  through it would make evaluate.py's tests fail for reasons that have
  nothing to do with evaluate.py. This does mean evaluate.py currently assumes
  callers pass something with a LightGBM-compatible `.predict()`, which is
  worth knowing about when model.py is implemented later.
- **cost_per_10k's threshold semantics**: chose `score >= threshold` (not
  `>`) as "flagged". Arbitrary but has to be picked consistently somewhere,
  and this is the natural reading of "at or above."
- The pr_auc "near base rate" test uses a fixed-seed (`default_rng(0)`)
  20,000-row uncorrelated random score vs. a 3.5% synthetic base rate, with an
  0.02 absolute tolerance picked before running the test rather than measured
  first. It passed on the first run (no correction needed), but that
  tolerance was a guess about how much a fixed-seed run of that size would
  wobble around the true base rate, not a derived bound — if this test ever
  gets flaky after a numpy/sklearn version bump, that tolerance is the first
  thing to revisit.

Full suite result (14 tests, all passing):

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\sohan\Desktop\not-strangers\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\sohan\Desktop\not-strangers
plugins: anyio-4.14.2
collecting ... collected 14 items

tests/test_data.py::test_raises_when_both_files_missing PASSED           [  7%]
tests/test_data.py::test_raises_when_identity_missing PASSED             [ 14%]
tests/test_data.py::test_raises_when_transaction_missing PASSED          [ 21%]
tests/test_evaluate.py::test_temporal_split_train_strictly_before_test PASSED [ 28%]
tests/test_evaluate.py::test_temporal_split_never_splits_a_tied_timestamp_group PASSED [ 35%]
tests/test_evaluate.py::test_temporal_split_rejects_invalid_test_size PASSED [ 42%]
tests/test_evaluate.py::test_pr_auc_perfect_ranking_is_one PASSED        [ 50%]
tests/test_evaluate.py::test_pr_auc_uncorrelated_scores_are_near_base_rate PASSED [ 57%]
tests/test_evaluate.py::test_recall_at_fpr_perfect_classifier_gets_full_recall PASSED [ 64%]
tests/test_evaluate.py::test_recall_at_fpr_inverted_classifier_gets_near_zero_recall PASSED [ 71%]
tests/test_evaluate.py::test_cost_per_10k_hand_computed PASSED           [ 78%]
tests/test_evaluate.py::test_cost_per_10k_is_parameterised_not_hardcoded PASSED [ 85%]
tests/test_evaluate.py::test_evaluate_model_returns_expected_keys PASSED [ 92%]
tests/test_evaluate.py::test_evaluate_model_accepts_threshold_and_cost_overrides PASSED [100%]

============================= 14 passed in 1.66s ==============================
```

## 2026-08-31 — Task 3: data check — stopping here

`data/train_transaction.csv` (and `data/train_identity.csv`) are not present
in this environment -- `data/` contains only `.gitkeep`. Per instructions,
skipping Task 4 (entities.py UID derivation) and Task 5
(`results/uid_validation.md` plus the size-distribution histogram) rather than
attempting a Kaggle download, since the Kaggle API credentials and
competition-rules acceptance aren't set up in this session.

Everything in Tasks 4 and 5 needs the real dataset to produce honest numbers
(row counts, uid label-purity, the actual distribution of transactions per
uid) -- there's no meaningful way to do them against synthetic data without
either fabricating the exact numbers the deliverable is supposed to report,
or building a synthetic-fraud-pattern generator to make the numbers
interesting, and both of those cut against what this session is for.

To pick this back up: run `bash scripts/download_data.sh` (needs
`pip install kaggle`, a Kaggle API token at `~/.kaggle/kaggle.json`, and
having accepted the competition rules -- see that script's header comment),
then re-run Task 4 and Task 5 from the original instructions.

## 2026-08-31 — entities.py implemented, uid_validation.md computed on real data

Kaggle credentials were set up between sessions and the real CSVs were
already present in `data/` (683MB train_transaction.csv, 26.5MB
train_identity.csv) -- no download needed this time.

Implemented `card1_addr1_origin_day` in `src/entities.py`. Changed
`resolve_entities`'s signature from the original scaffold stub
(`resolve_entities(transactions, key_columns)`, a generic union-find-style
design) to `resolve_entities(transactions)`, since the uid formula is now
fully specified and there's no longer a set of candidate key columns to
choose between -- keeping the generic parameter would have been dead
flexibility. 5 unit tests on a small hand-built frame (persistence across
days, discrimination on origin_day, all three null-cause cases, no dropped
rows), all passing.

Wrote `scripts/validate_uids.py` as a standalone analysis script (not wired
into run_pipeline.py or the Makefile) that reads only the 6 columns needed
from train_transaction.csv, computes every number in the spec, and writes
`results/uid_validation.md` and `results/uid_size_distribution.png` directly
from the computed values -- deliberately not hand-transcribed, to remove
transcription-error risk between the computed numbers and the report.

**Got wrong and corrected:** the script's first run crashed with
`pandas.errors.IndexingError: Unalignable boolean Series` at
`df.loc[uid.notna(), ["isFraud"]]`. `resolve_entities` returns a Series
indexed by TransactionID (as documented and tested), but `df` still had a
plain 0..n-1 RangeIndex from `read_csv`, so the boolean mask and the frame
didn't line up positionally by index label. Fixed by `df =
df.set_index("TransactionID")` immediately after computing `uid`, before any
`.loc` masking against it. Worth flagging because this is exactly the kind
of mismatch that silently produces wrong numbers instead of crashing when
the two objects happen to share a compatible-looking index -- here it
crashed loudly, which is the good outcome, but it's a real gotcha for any
other code that consumes `resolve_entities`'s output against the original
transactions frame.

**What surprised me:** two real findings from the actual data (reported
honestly, derivation NOT tuned to change either of them):

1. Fraud rate is very different between rows that got a uid and rows that
   didn't: **2.46%** for uid'd rows vs. **11.63%** for NaN-uid rows (see
   uid_validation.md section 5). Almost all of the NaN cases are addr1 being
   null (65,525 of 66,794), so this says addr1-missing transactions are
   substantially higher-risk than average, not just an artifact of the join. Any
   downstream cluster feature that silently excludes NaN-uid rows is
   therefore excluding a disproportionately fraud-heavy slice, not a random one.
2. `origin_day` (`day - D1`) is negative for **30.36%** of all transactions
   (mean -206.6, min -633), not just a rare edge case -- e.g. the largest
   impure uid in the report, `12695_325_-342`, has origin_day = -342. D1 is
   documented as "days since the card was first seen," so a negative
   origin_day means D1 exceeds the transaction's own elapsed-day count,
   which shouldn't happen if D1 and TransactionDT shared one consistent
   day-zero reference. It's a known-ish quirk of this dataset's D-columns
   (they don't reliably share TransactionDT's epoch), and it doesn't break
   the uid as an identifier -- the 98.53% unweighted purity result shows the
   card1+addr1+origin_day combination is still highly stable per persistent
   client -- but it does mean origin_day's literal value should not be read
   as a real calendar day. Not something to fix here per instructions (the
   derivation is exactly as specified); just flagging that the number's sign
   isn't meaningful on its own.

Label purity result: 98.53% of the 83,557 multi-transaction uids are fully
label-pure (unweighted), 97.61% weighted by transaction count -- the uid
derivation collapses card+address+cohort into a genuinely consistent-label
identity almost all of the time. Full numbers, the ten largest impure uids,
and the size-distribution histogram are in results/uid_validation.md.

## 2026-08-31 — D1/origin_day investigation (read-only, no derivation changes)

Follow-up on the negative-origin_day finding from the previous entry.
`src/entities.py` and `src/evaluate.py` untouched -- this was pure
investigation. Wrote `scripts/investigate_d1.py` -> `results/d1_investigation.md`.

**The root cause is now clear and it's benign.** Split by D1 within each
group tells the whole story: the "positive" group is 68.51% D1=0 (median
D1=0), i.e. mostly cards being seen for the very first time within the
dataset's own window, where origin_day trivially equals the transaction's
own day. The "negative" group has median D1=252 and max D1=640 -- these are
transactions from cards whose documented history (D1, "days since first
seen") already exceeds the entire ~year-plus span of TransactionDT covered
by this file. D1 isn't reset to this dataset's own day-zero; it reflects a
card's real prior history, which can predate the collection window
entirely. So negative origin_day isn't corrupted data, it's the arithmetic
consequence of joining an absolute "card age" feature against a
dataset-relative day counter. This matches last entry's guess ("D1 and
TransactionDT don't share one consistent day-zero") but now has the actual
mechanism behind it instead of just the symptom.

**Answering the four questions, briefly (full tables and numbers in
results/d1_investigation.md):**

1. Yes, negative-origin_day rows are a different population, not noise:
   lower fraud rate (1.60% vs 4.33%), much lower identity-record coverage
   (7.35% vs 31.97%), and a heavily skewed ProductCD mix (92.07% ProductCD=W
   vs 66.66%). They look like an older/more-established, lower-risk slice of
   traffic, consistent with "cards with a longer prior history."
2. D1 is not null-heavy in either group (can't be, by construction -- null
   D1 makes origin_day NaN, which is excluded from both groups entirely).
   It IS zero-heavy in the positive group (68.51%) and essentially never
   zero in the negative group (0%) -- this is the actual mechanism above.
3. Purity does NOT hold equally: 99.24% (weighted) for negative-origin_day
   multi-transaction uids vs 96.61% for positive. Per the task's own
   framing, that's the important case -- origin_day's sign is not cosmetic,
   it correlates with how trustworthy the resulting uid's label consistency
   is. Worth carrying forward: cluster features built on positive-origin_day
   (newer-card) uids are somewhat noisier than ones built on
   negative-origin_day (established-card) uids.
4. Collision check on the 20 largest uids: card2/card3/card5 are constant
   (1 distinct value) across literally all 20 -- expected, since they're
   sub-attributes of the same physical card as card1, not independent
   signals, so this adds nothing. P_emaildomain is where the real signal
   is: it varies (2-3 distinct values) in **10 of the 20** largest uids.
   card1+addr1+origin_day is genuinely merging multiple distinct people in
   half of the largest clusters -- confirmed by data, not assumed. This is
   "stability without correctness" exactly as the task named it: high label
   purity is not evidence the uid identifies one physical client, only that
   whoever it merges together tends to share a fraud outcome. Said plainly
   in the report; derivation was not touched in response to this.

**Nothing got corrected this time** -- the script ran cleanly on the first
attempt, most likely because the index-alignment gotcha from the previous
session (`df.set_index("TransactionID")` right after calling
`resolve_entities`) was already a known pattern going in, not something
rediscovered here.

## 2026-08-31 — Task 1: graph.py implemented, and a real methodology problem found and fixed

Implemented src/graph.py: LINKAGE_RULES (device_info, addr1_email,
card_bank_addr, each with a one-line rationale), build_entity_graph (hub
guard + edge construction, returns an EntityGraph(graph, excluded_hubs)
dataclass rather than a bare nx.Graph -- signature change from the scaffold
stub, needed to actually report what the hub guard excluded), get_connected_
components, and compute_cluster_features (10 columns: cluster size/txn
count/edge density/velocity/amount CV/burst concentration/email-uid ratio/
prior-fraud share, plus per-uid node degree and email domain count).
tests/test_graph.py: 10 tests, including the required explicit leakage test
(a future, huge-amount, fraud-labeled transaction for an already-clustered
uid must not change that uid's or its cluster's features at all -- verified
by asserting the feature frame is byte-for-byte identical with vs. without
that row).

**What surprised me, and it's a big one:** a 50k-row smoke test before
moving to Task 2 showed 29,211 of 30,185 nodes (96.8%) collapsing into ONE
connected component at the task's specified default, max_degree=1000. That
is not a ring, it's almost the whole active population -- cluster features
computed on it would be nearly constant across nearly every transaction and
carry no discriminating signal. Investigated properly on the full dataset
(524k uid'd rows) with a cheap union-find (no need to materialize edges to
check this): at max_degree=1000, the largest cluster is 127,708 uids (64% of
all 199,070 uids). Root cause: addr1 is a region-level code with only a few
hundred distinct values across the whole file, and card3/card5 are
dominated by one or two values (150.0/226.0 cover the overwhelming
majority) -- so card_bank_addr is nearly a proxy for addr1 alone, not an
independent signal, and even "below-hub" addr1-based groups (100-999
members) bridge transitively into one supercluster once enough of them
overlap.

Swept max_degree on the full dataset: 15 -> largest 77 uids; 20 -> 126;
25 -> 526; 30 -> 919; 35 -> **5,141**; 40 -> 6,625; 45 -> 10,074; 1000 ->
127,708. There's a sharp phase transition between 30 and 35 (a 5.6x jump),
not a gradual one -- this is a real structural property of the linkage
rules on this data, not noise. Also relevant for tractability: my clique-
based edge construction (itertools.combinations per shared-value group) is
O(k^2) per group; at max_degree=1000 a single near-threshold group can
contribute ~500k edges, and the 50k-row smoke test alone produced 6.4M
edges in 19s -- at full scale with the literal default this would likely be
tens of millions of edges and impractically slow. Lowering max_degree fixes
both problems at once.

**Fix:** kept build_entity_graph's own default at max_degree=1000, exactly
as specified in the task -- but run_pipeline.py calls it with max_degree=20,
documented prominently in graph.py's module docstring with the full sweep
so the number isn't just buried in a commit. Verified at full scale with
the real function (not just the union-find estimate): max_degree=20 builds
in 3.2s (199,070 nodes, 65,223 edges), connected components in 0.35s
(largest 126 uids), and compute_cluster_features over all 199,070 uids in
15.8s. This is a graph-construction/methodology decision, not a metric being
tuned -- the ablation lift hasn't been measured yet (that's Task 2) and this
choice was made before seeing any model result, purely from cluster-size
distributions.

**Got wrong and corrected (in the investigation, not in graph.py itself):**
my first attempt at investigating this on the real data returned "0 rows
covered" for every single linkage rule -- looked like a data problem, but it
was the exact same indexing bug class flagged in the entities.py DEVLOG
entry, just recommitted in a throwaway analysis script: I assigned
`df["uid"] = uid` without first calling `df.set_index("TransactionID")`, so
pandas aligned the RangeIndex against uid's TransactionID index and got
NaN almost everywhere. graph.py's own `_prepare()` helper already does this
correctly (that's precisely why it exists as a shared helper rather than
being repeated inline) -- the bug was only in my scratch investigation code,
not in the module being tested.

## 2026-08-31 — Task 2: model.py + run_pipeline.py implemented, ablation produced

model.py: train_baseline_model and train_cluster_model both call the same
private _fit() helper -- literally the same code path, same LGBM_PARAMS
dict (seed=42), same NUM_BOOST_ROUND=300 -- so the only way the two models
can differ is which columns are in X. build_feature_matrix drops isFraud
and a raw "uid" column if present, deliberately: the cluster model gets the
engineered cluster STATISTICS, never the raw uid string, so it can't just
memorize "this exact uid was fraud in training" instead of learning from
the aggregated signal -- that would inflate the ablation for reasons
unrelated to the cluster features actually being tested. 4 tests in
tests/test_model.py, including one that runs both training functions on
identical data and asserts the resulting boosters serialize identically
(`model_to_string()` equal) -- proving "identical treatment" structurally,
not just by eyeballing the params dict.

run_pipeline.py: load_and_prepare -> train_both_models -> evaluate_both_
models -> write_ablation_report, plus main(). Exposed as importable
functions (not just a __main__ script) so Task 3/4's scripts can reuse the
same data/graph/model artifacts without copy-pasting this orchestration.
Graph is built once from train-only data (max_degree=20, per Task 1's
finding) and cluster features are computed once with
as_of=test_df["TransactionDT"].min() -- the first test-period timestamp --
then broadcast from per-uid to per-TransactionID and left-joined onto both
train and test rows. Rows with no uid (~11%, see uid_validation.md) or a
uid with no pre-as_of history simply get NaN cluster columns via that
left-join -- never dropped, never a fabricated zero, exactly as instructed.
Policy.py is NOT called from here (out of scope this session, per
instructions) -- updated run_pipeline's module docstring accordingly since
the original scaffold-stub sequence included an apply_policy step.

**Result on the real full-scale temporal split** (472,432 train /
118,108 test rows, cost_fn=500, cost_fp=5, both illustrative and stated as
such in ablation.md):

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k |
|---|---:|---:|---:|
| baseline | 0.5646 | 0.4791 | 30,078.40 |
| cluster | 0.6322 | 0.5576 | 26,155.72 |

+0.0676 PR-AUC, +0.0785 recall@1%FPR, -3,922.68 cost per 10k. This is not a
small or marginal lift -- reported as-is, nothing was adjusted after seeing
it.

**What surprised me, and it needs Task 3 before anyone should believe it:**
`cluster_prior_fraud_share`'s feature-gain is 558,334 -- about 9x the
*second*-place feature (V258 at 63,550) and roughly 20x most others. That
single feature dominating this heavily is exactly the shape of a leak, even
though the unit-level leakage test in test_graph.py (future fraud-labeled
transaction must not change any feature) already passed. A passing unit
test on a small synthetic case proves the *mechanism* filters as_of
correctly; it doesn't by itself prove there's no leak at the *full-pipeline*
level (e.g. in how as_of is chosen, or in what "prior" means across the
train/test boundary in aggregate). Not fixing or second-guessing this
number now -- Task 3 is specifically the trace-it-and-report-plainly step,
next.

Runtime at full scale, for the record: ~35s to load data + build the graph
+ compute cluster features, ~58s to train both models. Cheap enough that
Task 3/4 can just re-run load_and_prepare()/train_both_models() fresh
rather than needing to cache artifacts to disk.

**Got wrong and corrected:** first cut of build_feature_matrix used
`select_dtypes(include="object")`, which raised a Pandas4Warning in
pandas 3.0.5 about "str" dtype columns being swept in implicitly under
"object" and that becoming an error in a future pandas version. Fixed by
being explicit: `include=["object", "str"]`. Caught by the model.py unit
tests before it ever touched the real 434-column dataset.

## 2026-08-31 — Task 3: adversarial sanity checks -- no leak found, but the lift is concentrated

Wrote scripts/sanity_checks.py, appending a "## Sanity checks" section to
results/ablation.md (not a separate file, since the re-ablation explicitly
needed to land as a second row in the same results table).

**1. Correlation with isFraud (train set):** cluster_prior_fraud_share is
0.7797 -- the only feature over the 0.5 red-flag threshold by a wide
margin (next highest is cluster_velocity at 0.057). This alone doesn't
prove a leak, but it's exactly the shape one looks like, so it drove
section 2.

**2. Traced cluster_prior_fraud_share for a leak -- found none, but only
after fixing my own check.** First attempt picked the first cluster with
*any* test-period fraud row as the demonstration example, and got a false
positive: reported, independently-recomputed, and "if-leaked" values all
came out identical (0.0563), which my comparison logic mislabeled as
"MISMATCH" because it required the leaked value to differ from the
reported one to count as a pass. Looking at *why* they matched: the
example uid already had fraud labels in the *train* period too, so its
per-uid "ever fraud" flag was already 1 before the test-period row existed
-- the test-period fraud couldn't have changed anything even if it had
leaked in, so the example proved nothing either way. That's a test-design
bug, not a pipeline bug, but it would have shipped a false "leak found" if
I'd trusted the first run's verdict text without reading the numbers.
Fixed by computing expected-vs-leaked prior-fraud share for *every*
cluster and picking the one with the largest gap as the example --
guarantees a non-vacuous demonstration instead of hoping for one. Also
added a global check across all clusters, not just the one example.

Real result: **0 mismatches across all 155,579 clusters** between the
pipeline's reported cluster_prior_fraud_share and an independent
recomputation from raw rows with `TransactionDT < as_of` (bypassing
graph.py entirely). The maximally-discriminating concrete example (cluster
#17894, a singleton uid with two fraud transactions, both in the test
period): reported = 0.0000, independently recomputed = 0.0000, what it
would be if test-period rows had leaked in = 1.0000. No leak.

**Why the correlation is still so high without leaking:** CLAUDE.md
already documents that labels "propagate across a card once reported."
cluster_prior_fraud_share is close to a direct measurement of exactly that
dynamic -- "has this persistent card-identity already been caught" --
which is fully legitimate as a *causal, non-leaking* feature, but its
predictive power comes largely from the same noisy label-propagation
process this project already flags as a caveat, not from discovering an
independent abuse signal. Worth remembering when reading the headline
lift number: high but not leaking is not the same as high and clean.

**3. Re-ablation without cluster_prior_fraud_share:**

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k |
|---|---:|---:|---:|
| baseline | 0.5646 | 0.4791 | 30,078.40 |
| cluster (full) | 0.6322 | 0.5576 | 26,155.72 |
| cluster (no prior_fraud_share) | 0.5756 | 0.4951 | 29,317.66 |

Removing that one feature shrinks the lift from +0.0676 to +0.0110 PR-AUC
(84% of the measured lift comes from that single feature), from +0.0785 to
+0.0160 recall@1%FPR, and from -3,922.68 to -760.74 cost per 10k. The
remaining lift is real, small, and driven by the other 9 structural/graph
features (edge density, velocity, burst concentration, email
heterogeneity, cluster size) -- reported as-is, not chased further to make
it look bigger.

**4. Cluster assignment independence from test-period edges: confirmed.**
0 graph nodes found with zero train-period transactions (should be 0, and
was). Concrete example: a uid appearing only in the test period is absent
from the graph and gets fully-null broadcast cluster features, exactly as
expected -- no train-period history means no cluster signal, not a
fabricated one.

**Bottom line for this task:** no leak found anywhere checked. The
headline ablation numbers from Task 2 stand, but the honest reading is
"~84% of the lift is one feature that's legitimate but rides on the
project's own documented noisy-label dynamic; the graph-structural
features alone add a smaller, genuine ~0.011 PR-AUC." Both numbers are now
in results/ablation.md; neither is hidden in favor of the other.

## 2026-08-31 — Task 4: threshold sweep and cost curve

scripts/cost_curve.py sweeps 300 thresholds (dense from 0-0.05, coarser
0.05-1.0, since the base rate is ~3.5% and both models' scores concentrate
near 0), plots both models' cost_per_10k curves, and marks each model's own
cost-minimizing point. Appends recall/FPR at each chosen point to
results/ablation.md.

**Got wrong and corrected (presentation, not numbers):** first version put
both "chosen point" annotations directly on a single full-range [0,1] plot.
Both models' optimal thresholds landed within 0.001 of each other
(0.0095 vs 0.0103) in the steep near-zero region, so the two text labels
overlapped into unreadable text. Fixed by splitting into two panels: the
full [0,1] range for shape, and a zoomed [0,0.05] panel (shaded on the
full-range plot to show where it comes from) with the annotations properly
offset apart. The underlying numbers were correct on the first run; only
the plot was unreadable.

**Chosen operating points:** baseline t=0.0095 (cost 30,009, recall 0.9326,
FPR 0.3813); cluster t=0.0103 (cost 25,923, recall 0.9530, FPR 0.3695). The
cluster model's cost-minimizing point is both cheaper AND has a lower FPR
at a higher recall -- a real, not cherry-picked, improvement here.

**Worth flagging plainly:** the cost-minimizing FPR is 37-38% for both
models. That's a direct, correct consequence of the assumed 100:1
cost_fn:cost_fp ratio (a missed fraud case is assumed 100x worse than one
false alarm, so the cost-optimal policy flags aggressively) -- not a bug,
but it means "cost-optimal" here means stepping up roughly a third of all
legitimate transactions, which is a real business-acceptability question
the illustrative cost ratio is doing all the work to answer. Said this
directly in ablation.md rather than only reporting the flattering
recall/cost numbers next to it.

## 2026-08-31 — Task 5: README results table and Limitations section

Filled in the Result table with the real numbers from all three model
variants (baseline, cluster, cluster minus cluster_prior_fraud_share),
linked to results/ablation.md for the full breakdown, and added a short
paragraph flagging that ~84% of the headline lift is one feature -- didn't
want the table alone to imply an uncomplicated win. Left "Why clusters",
"Data & labels", "Architecture" and "Running it" as empty headers: the task
asked specifically for the results table and a Limitations section, and
filling in the others wasn't asked for and isn't this session's call to
make.

Limitations section covers exactly the four things asked for: label noise
and its direct tie to cluster_prior_fraud_share's dominance; the uid
over-merging finding from d1_investigation.md (P_emaildomain varies in 10
of the 20 largest uids); the ~11% no-uid population and its much higher
fraud rate (11.63% vs 2.46%); and the no-ground-truth caveat that
everything measured here is feature lift on a fraud classifier, not
validated ring-detection accuracy.

Nothing surprising in this task -- it's a writing task pulling together
numbers already computed and verified in Tasks 1-4, not new analysis. No
corrections needed.

---

**Session summary (uid-and-graph branch, this multi-day session):**
graph.py, model.py and run_pipeline.py implemented; a genuine graph-
construction bug (giant-component collapse at the task's specified
max_degree default) found and fixed before it could contaminate anything
downstream; a real ablation run end-to-end on the full dataset; an
adversarial pass that found no leak but did find the lift is concentrated
in one feature, and said so; a cost curve with an honestly-reported
uncomfortable FPR at the "optimal" point; and a README that reflects all of
it, including the parts that don't make the result look as clean as a
single PR-AUC number would suggest. evaluate.py, entities.py, investigator.py
and policy.py were never touched, as instructed.

## 2026-08-31 — Task 1: investigator.py implemented

LLM layer over the Anthropic API (model claude-sonnet-4-6). explain_cluster
builds a flat evidence dict (build_evidence) from a cluster's member rows
of graph.compute_cluster_features output plus their raw transactions, sends
it to the model as a JSON code block (never prose) with a system prompt
whose hard rule is "never state a number not present in this JSON verbatim
or trivially rounded" -- stated explicitly as more important than the rest
of the prompt, since that's the one constraint this session actually checks
programmatically (Task 2, next). prioritize_clusters sorts by a simple,
clearly-labeled heuristic priority_score (prior-fraud share weighted
heaviest, txn volume and burst concentration as tie-breakers) -- explicitly
not a policy decision, no thresholds, no actions.

Graceful degradation: explain_cluster checks for ANTHROPIC_API_KEY before
attempting a call, and wraps the actual API call in a bare
`except Exception` -- either path returns a ClusterExplanation with
source="ungrounded-fallback" and a narrative built by directly formatting
the evidence dict's own values (grounded by construction, since it can't
contain anything that isn't already a dict value). Never raises either way.
6 tests, including one that monkeypatches `_call_anthropic` to raise and
confirms the fallback still returns cleanly.

**Important honest note for Task 2, which measures this module next:**
`ANTHROPIC_API_KEY` is NOT set in this environment. Every explain_cluster
call made in this session will take the fallback path. Task 2's
"groundedness" measurement will therefore be measuring the deterministic
fallback template (which is grounded by construction, since it's built by
formatting the evidence dict directly) -- not the actual LLM's behavior
under the prompt's hard-number rule. This is a real limitation of what this
session can demonstrate, not something to paper over: the LLM-calling code
path (_call_anthropic) is implemented and unit-tested with a mocked
success case, but has never actually been exercised against the real API
in this session. If a key is added later, re-running
scripts/eval_investigator.py is what would produce a real measurement of
the LLM's groundedness under the prompt.

## 2026-08-31 — Task 2: measured the investigator, confirmed the caveat from Task 1

scripts/eval_investigator.py -> results/investigator_eval.md. Selected 30
clusters spanning the risk range (all multi-uid clusters from the train
graph sorted by cluster_prior_fraud_share, 30 evenly-spaced percentile
points -- not just the top 30 riskiest, so low-risk and ambiguous clusters
are represented too, e.g. cluster-13 above with prior_fraud_share=0.0).
Groundedness check extracts every numeric literal from each narrative via
regex and checks it against the cluster's evidence dict, allowing rounding
to 0-4 decimals and percentage form (v cited as v*100).

**Result: 100% groundedness (0 of 360 extracted claims ungrounded), and
this number is exactly as uninformative as flagged in Task 1's DEVLOG
entry.** ANTHROPIC_API_KEY is still not set, so all 30 explanations took
the fallback path (source=ungrounded-fallback for every single one --
confirmed, not assumed, by checking the `sources` counter in the script's
output). The fallback template is grounded by construction (it's built by
directly formatting evidence dict values as `key=value` pairs), so 100%
was the only possible outcome here, not evidence that claude-sonnet-4-6
follows the hard-number-rule prompt. Said this plainly in
investigator_eval.md's first paragraph rather than reporting "100%
groundedness!" as if it validated the LLM.

What this task DID validate, honestly: the selection-and-measurement
*pipeline* itself works end to end -- risk-spanning cluster selection,
evidence assembly, narrative generation, numeric extraction, and grounding
comparison all ran correctly on real data (30 real clusters, real evidence
values, e.g. cluster-29's prior_fraud_share=1.0 with only 3 members and 7
transactions -- a small, high-confidence cluster). If a key is added, this
same script is what would need to be re-run for the number that actually
matters.

No corrections needed this task -- ran cleanly on the first attempt.

## 2026-08-31 — Task 3: policy.py + audit trail

Deterministic decide/apply_policy over allow/step_up/review. Thresholds
are NOT hand-picked: scripts/derive_policy_thresholds.py sweeps the
cluster model's real test-set scores twice, both at cost_fn=500, different
cost_fp: cost_fp=5 for step_up (25,923.31 min cost at t=0.010302 -- this
exactly matches Task 4's existing cost_curve.py output for the cluster
model, a good cross-check that both sweeps agree) and cost_fp=50 for
review (75,625.70 min cost at t=0.184343). The 10x cost_fp multiplier for
review vs step_up is stated as illustrative in policy.py's own docstring,
same caveat as every other cost assumption this project has made. Hard-coded
the two results as STEP_UP_THRESHOLD/REVIEW_THRESHOLD constants with the
script named as provenance, rather than recomputing them at import time.

decide() takes cluster_features but doesn't use it in the threshold
comparison (the policy is deliberately score-only, not cluster-conditioned)
-- documented explicitly in the docstring so this isn't mistaken for a bug,
and the parameter is kept because callers building an audit record want it
alongside the decision. apply_policy is a vectorized batch path proven
equivalent to calling decide() row-by-row (tests/test_policy.py), needed
for performance across the full entity population.

MODEL_VERSION is derived from model.py's own SEED/NUM_BOOST_ROUND constants
(`cluster_seed42_boost300`) rather than a hand-typed string, so it can't
silently drift from what actually trained.

build_audit_record assembles the required fields (transaction_id, uid,
model_version, score, threshold_applied, feature_values, action, reason,
timestamp) as a pure function; scripts/write_audit_sample.py does the
actual scoring/decisioning/writing, producing 200 real records in
results/audit_sample.jsonl from real test-period transactions (131 allow,
63 step_up, 6 review). Left several feature_values fields null in the
sample on purpose rather than filtering them out: some sampled
transactions have a uid with no pre-as_of cluster history (null cluster
features, exactly the "new-in-test-period" case established in Task 2 of
the previous session) and one sampled transaction has no uid at all --
that one still got a real decision (score 0.4364 -> review), which is
correct: the ~11% no-uid population is the highest-risk slice
(uid_validation.md) and must still be scored and decided on, never
silently skipped.

policy.py has no import of investigator.py -- checked two ways in
tests/test_policy.py: statically (parse policy.py's AST, assert no import
mentions "investigator") and behaviorally (apply_policy on identical
scores with ANTHROPIC_API_KEY set vs. unset AND
sys.modules["src.investigator"] simulated unavailable -- identical
DataFrames both times). 8 new tests total, all passing; 47 in the full
suite.

No corrections needed this task -- the two threshold sweeps and the audit
script all ran cleanly on the first attempt, likely because the sweep
methodology was already proven out in Task 4 of the previous session.

## 2026-08-31 — Task 4: calibration -- Brier score looked fine, the region that matters didn't

scripts/calibration.py: reliability curve (quantile-binned, 15 bins --
equal-width bins would dump almost everything into one bin given the
~3.5% base rate) plus Brier score, appended to results/ablation.md. No
retraining, no recalibration -- read-only measurement, as instructed.

**Got wrong and corrected before it shipped:** first pass computed a
single equally-weighted "mean absolute gap across bins" (0.0111) and used
`mean_abs_gap < 0.02` as the sole calibration verdict, concluding
"reasonably well calibrated." Looking at the actual plot before writing
that up: 14 of the 15 bins sit at very low predicted scores (below ~0.09)
where the model is naturally well-calibrated (predicting near-zero for a
mostly-zero outcome is the easy part), and the single highest-score bin --
mean predicted 0.4795, which is above BOTH policy.py thresholds
(STEP_UP_THRESHOLD=0.0103, REVIEW_THRESHOLD=0.1843) -- has an observed
positive rate of only 0.3509, a 0.13 gap. An equally-weighted average
across bins buries that one bad bin under fourteen easy ones. Rewrote the
verdict to be driven by the highest-score bin specifically (the region
closest to where the policy actually operates) rather than the blanket
average, and reported both numbers with the reasoning for why the average
is the wrong one to trust here.

**Verdict, correctly this time: not well calibrated where it matters,
despite a low Brier score (0.0200 vs. 0.0332 for a base-rate-only
predictor).** REVIEW_THRESHOLD should be read as an arbitrary cut on the
score scale, not "we estimate >18.43% abuse risk" -- scores up there
systematically overstate true risk. Noted isotonic/Platt scaling as the
fix, explicitly not implemented (model.py frozen, and the task said not to
retrain).

This is exactly the kind of mistake the whole session is structured to
catch -- a single scalar metric that looks fine in aggregate while hiding
a real problem in the one region that actually matters for the thing being
built. Caught here by looking at the plot before trusting the number, not
by a test (there's no unit test for "did I pick a misleading metric").

## 2026-08-31 — Task 5: latency and scale benchmark

scripts/benchmark.py measures three things separately and appends a
Performance section to ARCHITECTURE.md (previously empty since the
original scaffold): graph construction (2.67s for 472,432 train-period
transactions at max_degree=20 -- consistent with earlier ad-hoc timings
from Task 1's investigation), cluster feature computation (12.45s for
167,111 uids), and per-transaction inline scoring latency over 1,000 real
sampled test transactions (feature-store lookup + single-row
model.predict): p50=32.8ms, p95=37.3ms, p99=39.5ms.

Stated the batch/inline split plainly, as asked: graph construction and
cluster features are batch (never on the request path); scoring is inline,
assuming cluster features are already sitting in a lookup (simulated here
as a plain dict keyed by uid, built once from pipeline_data.cluster_features
-- not a pandas .loc on the full frame, since that's not what a real
feature-store read would look like).

Worth noting honestly: 32-37ms for a single LightGBM prediction is slower
than "just call predict()" might suggest, because the benchmark includes
the realistic cost of assembling a correctly-ordered, correctly-typed
442-column row (432 baseline + 10 cluster columns, several categorical) on
every call, including an explicit column reindex to guarantee exact
alignment with what the model was trained on -- this is genuinely part of
the inline cost, not overhead added by the benchmark, but a real system
would likely keep a pre-allocated schema rather than reassembling it per
request, which this number doesn't capture.

~1B transactions/quarter is ~2117x this benchmark's train set. Named three
concrete changes that scale would require, none implemented (this is a
performance write-up, not new engineering): incremental graph updates
instead of full rebuilds, approximate connected components, and sharding
across machines (with the real problem there being cross-shard edges, not
just storage). Also connected this forward to Task 1 of the previous
session's finding: max_degree's effect on cluster size is a phase
transition, not a smooth curve, which makes "just re-tune it at higher
volume" a riskier proposition than it sounds.

No corrections needed -- ran cleanly on the first attempt, including the
column-reindex safeguard added proactively before running rather than
discovered by a crash.

## 2026-08-31 — Task 6: cluster case studies

scripts/case_studies.py generates investigator explanations for all 1,567
multi-uid clusters in the train graph, ranks by priority_score, and prints
raw detail (per-uid breakdown, shared identifier values, amounts, time
span, fraud labels) for the top 3. results/case_studies.md is the manual
write-up on top of that raw output -- deliberately not templated, since
"does this look like a real ring" is exactly the judgment call a script
shouldn't make for you (said explicitly in the script's own docstring).

All 3 top clusters turned out to be small (2 uids each) and 100% fraud-
labeled -- not cherry-picked, it's a direct consequence of
priority_score weighting cluster_prior_fraud_share at 100x: a small
cluster where every known transaction is fraud maxes that term out, so
small-and-fully-fraudulent clusters dominate this particular ranking.
Worth stating about the ranking itself, not just the 3 clusters it
surfaced.

Checked which specific linkage rule created each cluster's edge by reading
the actual `rules` edge attribute from the graph object (not inferred from
shared-value tables) -- this mattered:
- **Case 1** (`card_bank_addr`, an uncommon card3=223/card5=224 pairing --
  most transactions in this dataset cluster on card3=150/card5=226, so
  this specific combination matching is rarer and more specific than the
  hub-guard's raw uid-count threshold alone would suggest): plausible true
  positive.
- **Case 2** (`device_info`, a specific Samsung build string
  `SM-G950F Build/NRD90M` linking two otherwise-unconnected card+address
  pairs, transactions 4.7 minutes apart, identical $300 amounts, both
  fraud): the strongest case of the three.
- **Case 3** (`device_info`, but the shared value is `"en-gb"` -- reads as
  a locale/language string, not a device fingerprint): flagged explicitly
  as the weakest case, per the task's instruction to include a
  possible-false-positive and say so rather than swap it for a
  cleaner-looking example. The behavioral pattern (3.7 minutes apart,
  identical $150 amounts, both fraud) is genuinely suspicious and mirrors
  Case 2's shape, so it isn't confidently a false positive either --
  reported as ambiguous, with the specific reason the identifier is weaker
  spelled out (a value real unrelated en-GB-locale users could plausibly
  share, only below max_degree=20 by chance in this data slice, not
  because it's actually rare).

This is the first task this session where the deliverable is a judgment
call rather than a number -- worth being honest that "plausible true
positive" and "ambiguous" are reads, not measurements, and the report says
so in its own closing section rather than implying more precision than 3
manually-inspected clusters out of 1,567 can support.

## 2026-08-31 — Task 7 (part 1): wired everything into run_pipeline.py

Moved the artifact-generation logic from scripts/sanity_checks.py,
cost_curve.py, calibration.py, eval_investigator.py, write_audit_sample.py
and benchmark.py into src/run_pipeline.py as reusable functions
(write_sanity_checks, write_cost_curve, write_calibration,
write_investigator_eval, write_audit_sample, write_benchmark), all called
from one main() after a single load_and_prepare()/train_both_models() pair
-- previously each of the 7 artifacts independently re-loaded the 683MB
CSV and re-trained both models, ~90s of pure redundant setup per script.
Each scripts/*.py file is now a thin wrapper delegating to the
corresponding run_pipeline function, kept for standalone use (`python
scripts/cost_curve.py` still works).

**Verification that nothing's numbers changed in the move:** backed up the
committed results/ artifacts, ran the new consolidated `python -m
src.run_pipeline`, and diffed. results/ablation.md: byte-identical.
results/cost_curve.png, results/calibration.png: identical MD5. 
results/investigator_eval.md: one line differs, purely cosmetic wording
("Re-run this script" -> "Re-run this", since the text moved from a
script's own docstring into a shared function that isn't only reachable
from a script anymore) -- zero numbers changed. results/audit_sample.jsonl:
0 of 200 records differ once the timestamp field (which is supposed to
differ -- it's wall-clock time of the run) is excluded from the
comparison.

**Got wrong and corrected before committing:** results/ablation.md's
sections are safe to re-append repeatedly because write_ablation_report
always regenerates the whole file fresh first -- but ARCHITECTURE.md has
no equivalent "start fresh" step, and write_benchmark was a straight
unconditional append. Running the consolidated pipeline once revealed
`ARCHITECTURE.md` had gained a SECOND "## Performance" section (grep count:
2). This is exactly the kind of bug the reproducibility check in part 2 of
this task is supposed to catch -- `make results` needs to be safe to run
more than once, and it wasn't. Fixed with a `_remove_section()` helper that
strips a heading through to the next top-level heading before
re-appending, applied to both the ARCHITECTURE.md path and (defensively)
`_append_to_ablation`'s per-section appends. Verified the fix by running
the full pipeline twice in a row and confirming exactly one "## Performance"
section and one each of the three ablation.md subsections both times, with
identical model metrics both runs.

## 2026-08-31 — Task 7 (part 2): reproducibility verification

Cloned the uid-and-graph branch (`git clone --branch uid-and-graph
--single-branch`) into a fresh temp directory, created a brand-new venv
inside the clone, and `pip install -r requirements.txt` from scratch --
all packages installed cleanly at the exact pinned versions with no
resolver conflicts (same pins verified back in the very first scaffolding
session: pandas 3.0.5, numpy 2.5.0, scikit-learn 1.9.0, lightgbm 4.6.0,
networkx 3.6.1, matplotlib 3.11.1, shap 0.52.0, anthropic 0.120.2,
pytest 9.1.1).

**One step failed and is worth reporting plainly: `make results` does not
run on this machine, because GNU Make isn't installed at all** (`which
make` finds nothing on this Windows/Git-Bash setup). The README's
"Reproduce: `make results`" line assumes a Unix-like environment with make
available (a dev container, CI, macOS/Linux) -- on a bare Windows checkout
without WSL or make installed via a separate package manager, a reviewer
hits this immediately. Since the Makefile's `results` target is exactly
`python -m src.run_pipeline`, that command was run directly instead, which
is what actually verifies the pipeline -- but the `make` wrapper itself
was not exercised, and this is a real gap for a Windows-only reviewer, not
a hypothetical one.

data/ is (correctly) empty in a fresh clone -- CLAUDE.md's "do not commit
anything under data/" rule working as intended, but it also means a fresh
clone cannot run the pipeline until the two CSVs exist locally. Copied
train_transaction.csv and train_identity.csv from the already-downloaded
copy in the main working directory into the clone's data/ rather than
re-running scripts/download_data.sh against Kaggle again -- re-testing the
live Kaggle download wasn't the point of this check (it was already
exercised when the data was first fetched, see the entities.py session's
DEVLOG entry) and would have added a real external dependency (valid
credentials, competition rules accepted, network access) to a check that's
really about the code and environment being reproducible. Worth being
explicit that this means the download step itself was not re-verified here.

With data/ populated: `pytest tests/` -- **47 passed**, matching the
working directory exactly. `python -m src.run_pipeline` -- **completed
successfully**, and every number matched the working directory's committed
results exactly: results/ablation.md byte-identical, results/cost_curve.png
and results/calibration.png identical MD5 hashes, baseline/cluster
PR-AUC/recall/cost all identical to the last decimal printed
(0.5646104419855579 / 0.6322404738191845 for PR-AUC, matching bit for bit).

**Summary: the pipeline itself is fully reproducible from a fresh clone,
fresh venv, and fresh pip install, given the two source CSVs are present.
The one broken step is `make` not existing on this Windows environment** --
not a code bug, but a real onboarding gap between what the README promises
and what a plain Windows checkout can do without additional setup (installing
make, or just running `python -m src.run_pipeline` directly as documented
here). Not fixed this session (installing system-level `make` is outside
what a code change can do, and rewriting the Makefile as something
Windows-native wasn't asked for) -- reported plainly instead, per this
session's own instructions.

## 2026-08-31 — A silent failure hid an entire unmeasured layer, and my own eval reported it as a success

This is the most important bug this project has produced, and it wasn't
caught by any test, any sanity check, or any of the "measure it, don't
just ship it" discipline this session has otherwise followed. It was
caught by the user reading results/investigator_eval.md and noticing that
"ANTHROPIC_API_KEY was set" and "30/30 explanations were
source=ungrounded-fallback" were both true on the same line, and asking
why.

**Root cause, as diagnosed and fixed by the user, not by this session's
own investigation:** the Anthropic API key in use is identity-linked (a
workspace-scoped key), which the API rejects on every single call unless
the request carries an `anthropic-workspace-id` header. investigator.py
never sent that header. Separately, the workspace's balance was zero.
Both are now fixed on the account side; the header support is fixed here.

**Why this is a real bug class, not just a one-off missing header:** this
project's own design -- correctly -- required investigator.py to degrade
gracefully: no API key, a network blip, a rate limit, anything, and
`make results` must still complete rather than crash. That's the right
call, and it's still the right call after this bug. The actual defect was
one specific line: `except Exception: return <fallback>`, with nothing
recording *why*. A graceful degradation path and a silent failure path are
the same code until you ask "does anything downstream know the difference
between 'degraded on purpose' and 'broke and I didn't notice'?" Here,
nothing did. Every one of 30 calls failed with the exact same
authentication error, for an entire session's worth of work (Task 1's
implementation and Task 2's "measurement"), and the code had no way to
surface that -- not a log line, not a counter, not a flag. The fallback
path being *correct* is precisely what made the failure invisible: a
crash would have been noticed in the first five seconds.

**The compounding failure, which is the part worth sitting with:** Task 2
of the earlier session was explicitly framed as "measure the investigator,
don't just ship it" -- and it did report 100% groundedness with a caveat
that the key was unset. That caveat was correct *then*. But this most
recent run had a key set, and the report-writing code
(write_investigator_eval, then in scripts/eval_investigator.py) contained
this: the "at least one explanation used the real LLM path" sentence was
selected by `if not has_key: ... else: "at least one explanation used the
real LLM path"` -- driven by whether a key was *present*, not by whether
`sources` actually contained anything other than `ungrounded-fallback`.
The measurement layer -- the thing built specifically to catch exactly
this kind of problem -- inherited the same category of bug as the thing it
was measuring: it asserted an outcome from a precondition, not from the
actual result. A precondition ("a key was set") is necessary but not
sufficient for success, and reporting code must never conflate the two.
This is the generalizable lesson: any report about whether a fallback path
was needed must be computed from the actual recorded outcome of every
attempt, never inferred from whether success was theoretically possible.

**What was fixed, concretely:**
- `investigator.py`: `_call_anthropic` now reads `ANTHROPIC_WORKSPACE_ID`
  from the environment and passes it as `default_headers=
  {"anthropic-workspace-id": ...}` when set, omitted entirely when unset
  (a non-identity-linked key keeps working unchanged -- verified with a
  test that inspects the actual kwargs a mocked `anthropic.Anthropic` was
  constructed with, for both the header-present and header-absent cases).
- `ClusterExplanation` gained an `error: str | None` field: `None` on a
  real LLM success, `"ANTHROPIC_API_KEY not set"` or `"{ExceptionType}:
  {message}"` on every fallback. `explain_cluster` now also prints
  `[investigator] {cluster_id}: ...` to stderr on every fallback, so a
  `make results` run has a visible trail even if no one reads the written
  report afterward. The fallback *behavior* is unchanged -- still never
  raises, still completes -- only its visibility changed.
- `run_pipeline.write_investigator_eval` rewritten so every claim in
  results/investigator_eval.md is derived from `sources`/`error`, never
  from `has_key`: it now says "0 of 30 explanations used the real LLM
  path" when that's true regardless of whether a key was present, and adds
  a "Fallback errors encountered" section listing every distinct error
  string and how many clusters hit it. Groundedness is now reported
  separately for LLM-sourced vs. fallback-sourced narratives (a mix would
  otherwise dilute or inflate the number that actually matters), with an
  explicit "there is nothing here that measures claude-sonnet-4-6's actual
  behavior" line when n_llm=0, and a defensive check that flags (rather
  than silently trusts) an ungrounded claim ever showing up in a fallback
  narrative, since that would itself be a bug given the template is
  grounded by construction.
- 7 new tests: `error` populated correctly for both fallback causes and
  left `None` on success, the stderr log line contains the cluster id and
  the real exception type/message, and the workspace header is present
  only when `ANTHROPIC_WORKSPACE_ID` is set. 54 tests passing overall.
- README.md documents `ANTHROPIC_WORKSPACE_ID` under a new "Running it ->
  Environment variables" section, including the specific symptom (looks
  like it's working, 100% fallback) so a future reader recognizes this
  failure mode immediately instead of re-discovering it the way this
  session did.

**What this entry cannot honestly claim:** I do not have
`ANTHROPIC_API_KEY` or `ANTHROPIC_WORKSPACE_ID` available in my own tool
execution environment -- checked Bash, PowerShell, persisted Windows
user/machine environment variables, and for a `.env` file; none were
found. Re-running `scripts/eval_investigator.py` here still produces 0 of
30 real LLM explanations, now *correctly* reported as
`ANTHROPIC_API_KEY not set` rather than incorrectly implying success. The
user has verified a direct API call succeeds in their own environment, but
I have not personally produced or witnessed a real groundedness
measurement against actual LLM output, and I am not reporting one. That
measurement still needs to happen in an environment where the credentials
are actually reachable -- this entry fixes the mechanism for measuring it
honestly; it does not itself contain that measurement.

## 2026-08-31 — The real measurement: 100% groundedness on actual LLM output

The user ran `scripts/eval_investigator.py` themselves, in an environment
where `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` are both genuinely
set and working, and shared the resulting results/investigator_eval.md.
This is the measurement the previous two entries explicitly said had not
happened yet -- recorded here as soon as it existed, not reconstructed or
approximated.

**Result: 30 of 30 explanations used the real LLM path (source=llm,
confirmed from the actual `sources` counter, not inferred). 0 of 182
extracted numeric claims were ungrounded -- groundedness rate 100.00%.**
Zero fallback explanations this run, so unlike every previous run, there's
no fallback-vs-LLM split to worry about diluting the number: all 182
claims are real claude-sonnet-4-6 output.

Spot-checked the three example narratives by hand against their evidence
dicts rather than just trusting the automated checker's own arithmetic:
cluster-13 cites edge density 1.0, email-uid ratio 0.1429, 1 distinct email
domain, 2 distinct product codes, amt_cv 1.0557, prior_fraud_share 0.0,
burst_concentration 0.0909 -- every one matches its evidence dict value
exactly. cluster-29 cites prior_fraud_share 1.0, edge_density 1.0, 2
product codes, 2 email domains, email_uid_ratio 0.6667 -- same, all exact
matches. The narratives also read as genuinely reasoned rather than just
regurgitating the evidence dict in prose: cluster-13's read ("structural
linkage and email concentration are the most concerning signals... though
the absence of prior fraud history and low burst activity leave the risk
assessment somewhat ambiguous") explicitly weighs signals against each
other rather than asserting a flat verdict, which is exactly the kind of
grounded-but-not-mechanical behavior the system prompt's hard rule was
meant to allow room for.

**What this does and doesn't establish.** It closes the loop the last two
entries left open: the investigator layer has now actually been exercised
against the real model, not just unit-tested against mocks, and the
specific hard-number-grounding constraint this session was built around
held on every single claim it produced. It does not establish that
claude-sonnet-4-6 will always stay grounded -- this is one run, 30
clusters, 182 claims, not a guarantee, and the honest posture from here is
"measured clean on the one real run we have," not "proven safe." If this
is re-run at higher volume or on a different cluster selection, the
groundedness check should keep running every time, not be treated as a
box now permanently checked.

This also means the bug fixed two entries ago is now confirmed fixed
end-to-end, not just plausible in theory: the workspace-id header was the
actual blocker, and with it in place the same code path that produced 30
silent failures now produces 30 real, grounded explanations.

## 2026-09-01 — Task 1+2: Streamlit dashboard (app.py) + dark theme

Committed together rather than split: the CSS is embedded directly in
app.py (there's no separate "logic" version that ever existed without
styling), so artificially splitting them into two commits would mean
reconstructing a fake unstyled intermediate just for git history's sake,
which helps no one. Both tasks are genuinely done, documented together
honestly.

Three tabs: Review queue (cluster list ranked by investigator.py's
priority_score, selectable, with a detail panel showing the LLM narrative,
evidence table, member uids, a transaction timeline, and the policy
decision + threshold), Model performance (ablation table, cost curve,
calibration -- all parsed from results/ablation.md at runtime), Audit
trail (results/audit_sample.jsonl, filterable by action). Dark theme via
.streamlit/config.toml (near-black #0D0D0F base) plus custom CSS injected
via st.markdown for anything the theme config can't reach (fonts,
monospace numerics, risk badges).

**Real problems found and fixed while building this, in the order they
were hit:**

1. **Queue-building took minutes, not seconds.** First version looped
   `investigator.build_evidence` + `_priority_score` + `policy.decide`
   over every one of the ~1,500+ multi-uid clusters with test-period
   activity, each iteration doing a `.loc` fancy-index lookup against a
   ~199k-row cluster_features table. Caught by literally watching it hang
   for minutes with the server still alive and the log not moving.
   Rewrote as two passes: a fully vectorized pass (groupby `.count()`,
   `.map()` against a Series -- no per-cluster Python callback) to find a
   bounded candidate pool of 400 by a cheap proxy
   (`cluster_prior_fraud_share`, the dominant term in the real priority
   formula anyway), then the exact, non-vectorizable investigator/policy
   calls only on that bounded pool. Seconds instead of minutes, same
   numbers for every cluster that ends up in the displayed queue.

2. **A visibly empty bordered box between the policy-decision line and
   the LLM narrative.** Root cause: `st.markdown('<div class="panel">')`
   opened in one call, with the label/content/closing `</div>` issued in
   *separate* `st.markdown` calls after it. Each `st.markdown` call is its
   own independent HTML fragment in Streamlit -- an unclosed `<div>` in
   one call does not wrap content from a later call, so the "open" div
   rendered as its own empty, styled, bordered rectangle, and the actual
   content became unrelated sibling elements that happened to look
   plausible next to it. Fixed by building the whole panel's HTML as one
   f-string and issuing it in a single `st.markdown` call. Checked the
   rest of the file for the same pattern (grep for every `st.markdown`
   call with raw HTML) -- this was the only instance.

3. **A hardcoded number, caught by looking at my own screenshot.** The
   Model performance tab's caption said "roughly 84% of the headline
   PR-AUC lift (+0.0676 full vs. +0.0110 without it)" as a literal string
   -- correct today, but exactly the kind of thing this session's own
   instruction ("never invent a number... every figure must be read from
   results/ at runtime") was written to prevent, since it would go
   silently stale if the underlying numbers ever changed. Fixed to compute
   the share and both deltas from the already-parsed ablation table at
   render time.

4. **Streamlit's rerun-vs-runOnSave behavior isn't what I assumed.**
   While testing, I edited app.py *while a capture run was already
   executing* against the previous version, expecting the running
   subprocess to keep using the old in-memory code (`runOnSave=false` in
   config.toml). It didn't: a widget-triggered rerun (clicking a queue
   row) picked up the edited file mid-run, producing a screenshot with
   the new styling despite being launched before the edit. Streamlit
   re-reads the script from disk on every rerun regardless of
   `runOnSave` -- that setting only controls whether the file watcher
   *automatically* triggers a rerun without user interaction (showing a
   "File change / Rerun" banner instead). Didn't affect correctness here,
   but worth knowing: don't trust that a long-running dev-server subprocess
   is still running the code you started it with.

Chose a hand-rolled HTML table (`render_html_table`) over `st.dataframe`
for the evidence/ablation/audit tables specifically to get monospace,
right-aligned, consistent-decimal numerics -- `st.dataframe`'s grid
(glide-data-grid) renders to canvas and can't be reached by CSS at the
per-cell level. `st.dataframe` is used only for the one place true
row-selection interactivity is required (the queue), with a pandas Styler
for the risk-colored action column (`.style.map`, confirmed it renders
correctly rather than assumed).

Verified live with a real running server + Playwright screenshots at each
step (not just "it compiles") -- all three tabs, row selection, the
narrative/policy attribution, the action-column coloring, and the
error/empty states were each checked against an actual screenshot before
moving on, per this session's own instruction to test UI changes in a
browser rather than claim success from type-checking alone.

## 2026-09-01 — Task 3: screenshots

scripts/capture_screenshots.py launches `streamlit run app.py` as a
subprocess on a scratch port, waits for the one-time pipeline load, and
uses Playwright (headless Chromium) to capture the review queue, a
cluster detail view (first queue row selected), and the model performance
tab into docs/, then tears the subprocess down. Not wired into `make
results` or requirements.txt -- this is a one-off authoring tool for the
README, not something the pipeline or dashboard needs at runtime;
playwright/chromium were installed ad hoc for this and for live-testing
app.py during Task 1+2, documented in the script's own docstring rather
than silently added as a project dependency.

Screenshots are real: same server, same real data, same real LLM/fallback
behavior as every other verification this session -- not mocked or
hand-assembled.

No corrections needed for the script itself -- it worked on the first
run (the styling fix from Task 1+2 was still in flight when this was
first written, so the first capture predated it slightly; a second run
after that edit landed produced the final, correct screenshots, confirmed
by inspecting all three images before committing).

## 2026-09-01 — Task 4: README rewrite -- results first

Restructured per the session's spec: one-sentence description, results
table with the 84%-lift disclosure stated before the reader gets to
celebrate the headline number, both reproduce commands (`python -m
src.run_pipeline` and `make results`, with the Windows caveat about make
not being present stated plainly rather than assumed away), setup
(venv, data download, both Anthropic env vars with the workspace-id
failure mode named explicitly so a future reader recognizes the symptom
instead of re-discovering it), an architecture summary that states the
ML/LLM/policy separation as a structural fact with its own verification
(AST check + behavioral test), screenshots, and all seven limitations
the task asked for expanded (label noise's interaction with the dominant
feature, uid over-merging, the ~11% unresolvable rows and their higher
fraud rate, max_degree=20 and what it excludes, calibration
overconfidence of 0.13, no ground truth, and the groundedness result
being one run of 30 clusters, not a permanent guarantee).

Every number in it was copied from an already-verified results/ artifact
or DEVLOG entry, not recomputed or re-derived for this rewrite -- nothing
new to check here beyond making sure the transcription matched the
source exactly, which it does (diffed by eye against results/ablation.md,
results/investigator_eval.md, results/uid_validation.md, and
results/d1_investigation.md while writing).
