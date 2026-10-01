"""Anonymizer evaluator: compares the output of a system with the ground truth.

The metric is defined in ``docs/metrics.md``. Programmatic entry point::

    from test_bench.evaluation import evaluate
    result = evaluate(manifest, report, output_dir)

and from the command line ``uv run python -m test_bench.evaluate``.
"""

from test_bench.evaluation.core import ElementEval, FileEval, MetadataEval, evaluate, evaluate_file

__all__ = ["FileEval", "ElementEval", "MetadataEval", "evaluate", "evaluate_file"]
