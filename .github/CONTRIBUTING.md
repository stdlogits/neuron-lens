# Contributing

[AGENTS.md](../AGENTS.md) is the guide for people and coding agents alike: the layout, the
invariants a change must keep, how to run the pipeline, and how to record a result. Read
[docs/findings.md](../docs/findings.md) before making a claim about a neuron or a lens.

Contributions come as branches and pull requests. CI runs the tests on CPU; when a change touches
the CUDA kernels, also run `tests/test_kernels.py` on a CUDA machine.
