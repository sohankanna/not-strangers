# Project context
Detecting coordinated payment abuse rings for the Razorpay AI Buildathon.
Data: IEEE-CIS Fraud Detection. Labels are chargeback-reported and propagate
across a card once reported — treat as noisy.

# Rules
- Splits are ALWAYS temporal on TransactionDT. Never random.
- Never modify evaluate.py to improve reported numbers.
- No synthetic fraud/abuse generation of any kind (track is defense-only).
- The LLM layer explains and prioritizes. policy.py decides. Never merge them.
- Do not commit anything under data/.
- Cluster features must be computed causally. Graph structure and aggregates
  for a test transaction may only use transactions with strictly earlier
  TransactionDT.
- test_transaction.csv has no labels (Kaggle competition holdout, never
  released). All train/test splitting happens INSIDE train_transaction.csv,
  sorted by TransactionDT, last 20% held out.
- Constructing arrays of synthetic labels and scores for unit-testing metric
  functions is fine and is not fraud generation. The defense-only rule
  prohibits generating realistic abuse/fraud patterns or attack tooling, not
  test fixtures like np.array([0,1,1,0]).

