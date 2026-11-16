# Demo

`run_demo.py` runs the full decision layer on synthetic data.

```bash
python demo/run_demo.py
```

What it shows: conformal intervals and RPD grading, budget-constrained
routing under three loss structures, the guarded agent (rule path when no
API key), the multi-agent deliberation interface, report rendering,
validation, feedback selection, and the degradation table.

The upstream model is simulated by a per-view ridge regression; only the
decision layer is under test here.
