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

