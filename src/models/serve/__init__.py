"""Serving / inference layer: one module per trained track plus the unified
``inference`` facade. Each module loads a persisted model and maps raw postings
to predictions (no fitting). The training counterparts live in
``src.models.train`` and ``src.models.tracking``.
"""
