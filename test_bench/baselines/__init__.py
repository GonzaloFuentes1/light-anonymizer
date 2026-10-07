"""Baselines that validate the evaluator.

Each module exposes ``process(file_entry, manifest, folder, details) -> FileResult``:

- ``file_entry``: the manifest record (``schema.FileEntry``).
- ``manifest``: the full manifest (its ``root`` is the folder of the input files).
- ``folder``: the ``<output>/files`` folder; the output file goes to ``folder / file_entry.path``.
- ``details``: free-form dictionary that ends up in ``RedactionReport.details`` (supporting information).

Iterating over all files, timing and writing the report live in ``test_bench.baseline``.
"""

SYSTEMS = ("identity", "oracle", "notebook", "prototype")
