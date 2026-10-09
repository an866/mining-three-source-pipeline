"""Evaluation package: ground truth, metrics, and the harness that joins them.

Kept a package rather than a loose script so the ground truth and the metrics can be
unit-tested without loading an index, and so ``python -m eval.harness`` works from
the repository root with no path juggling.
"""
