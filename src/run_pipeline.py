"""CLI entry point for `make results`. The only orchestration layer.

Sequences entities -> graph -> model -> evaluate -> policy. None of those
modules should grow their own __main__/CLI code; this is the one place that
wires them together, so evaluate.py in particular can stay a pure metrics
module (see its module docstring).
"""

from __future__ import annotations


def main() -> None:
    """Run the full pipeline and report results.

    Intended sequence:
      1. Load train/test transactions and resolve entities
         (entities.resolve_entities).
      2. Build the entity graph and compute causal cluster features for each
         split (graph.build_entity_graph, graph.compute_cluster_features),
         using the temporal split from evaluate.temporal_train_test_split to
         set `as_of` correctly for the test period.
      3. Train the baseline and cluster-augmented models
         (model.train_baseline_model, model.train_cluster_model).
      4. Evaluate both models with evaluate.evaluate_model and report the
         results table promised in README.md.
      5. Apply policy.apply_policy to the cluster-model scores to produce
         final decisions.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
