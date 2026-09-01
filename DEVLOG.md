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
