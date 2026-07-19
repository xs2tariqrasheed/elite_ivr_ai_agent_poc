"""Evaluation harness for the quick_reservation ("Ann") agent.

This package is built up phase by phase as a learning exercise:

  Phase 0  harness.py   drive the agent headless, capture transcript + snapshot
  Phase 1  scorers.py   deterministic (code-based) scoring
  Phase 2  dataset.py   the test cases
  Phase 3  judge.py     LLM-as-judge for subjective quality
  Phase 4  caller.py    an LLM that role-plays the caller
  Phase 5  run.py       one command that prints a scorecard

Everything is plain Python with no test framework, matching the existing
standalone style of tests/test_barge_in.py. Run any file directly, e.g.:

    python evals/harness.py
"""
