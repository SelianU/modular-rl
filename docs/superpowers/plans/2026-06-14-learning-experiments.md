# Learning Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runnable learning experiments that train each model on slightly richer toy tasks with multiple parameter settings and detailed logs.

**Architecture:** Keep pytest learning tests as fast correctness checks. Add a separate experiment runner under `experiments/learning/` for longer, parameterized runs, plus a shell wrapper under `scripts/` so beginners can execute everything with one command.

**Tech Stack:** Python, PyTorch, pytest for validation, Bash wrapper scripts.

---

### Task 1: Experiment Runner Contract

**Files:**
- Create: `tests/learning/test_learning_experiments_runner.py`
- Create: `experiments/learning/run_model_learning_experiments.py`

- [ ] **Step 1: Write the failing tests**

Add tests that import `build_experiment_specs`, filter experiments by model name, run a tiny subset, and verify metric dictionaries include loss, accuracy, and parameter metadata.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/learning/test_learning_experiments_runner.py -v`

- [ ] **Step 3: Implement minimal runner API**

Create dataclasses for experiment specs and results, model/task builders, training helpers, filtering, and summary printing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/learning/test_learning_experiments_runner.py -v`

### Task 2: Beginner Shell Wrapper

**Files:**
- Create: `scripts/run_learning_experiments.sh`
- Modify: `README.md`

- [ ] **Step 1: Add wrapper**

Create a Bash wrapper that chooses `.venv/bin/python` when available and forwards CLI arguments.

- [ ] **Step 2: Document usage**

Document `./scripts/run_learning_experiments.sh`, `--model`, `--quick`, and example log output.

- [ ] **Step 3: Verify commands**

Run the quick full experiment set and a single-model filtered run.

### Task 3: Final Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run targeted tests**

Run: `.venv/bin/python -m pytest tests/learning/test_learning_experiments_runner.py -v`

- [ ] **Step 2: Run existing tests**

Run: `.venv/bin/python -m pytest -v`

- [ ] **Step 3: Run experiment scripts**

Run: `./scripts/run_learning_experiments.sh --quick` and `./scripts/run_learning_experiments.sh --model mlp --quick`.
