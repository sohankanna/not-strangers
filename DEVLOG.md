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
