# Naming Consistency Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make variable, helper, and example names consistent without breaking the public API.

**Architecture:** Preserve public class and function names such as `DQNAgent`, `build_trainer`, `make_mlp`, and `quick_dqn`. Rename internal/local variables and example helper names to explicit `snake_case` forms, then verify behavior with the existing pytest suite.

**Tech Stack:** Python 3.8+, PyTorch, Gymnasium, pytest.

---

### Task 1: Baseline Verification

**Files:**
- Test: `tests/test_config_builders.py`
- Test: `tests/test_simple_networks.py`

- [ ] Run `.venv/bin/python -m pytest -v`.
- [ ] Confirm the existing suite passes before renaming.

### Task 2: Internal Builder and Factory Names

**Files:**
- Modify: `modular_rl/training/builders.py`
- Modify: `modular_rl/training/factory.py`
- Modify: `modular_rl/training/logger.py`

- [ ] Rename local `ctx` to `context`.
- [ ] Rename `q_net` to `q_network`.
- [ ] Rename `actor_bb`, `critic_bb`, and similar local names to `actor_backbone`, `critic_backbone`.
- [ ] Rename `cfg` to `backbone_config` or `model_config`.
- [ ] Rename `lg` to `logger`.

### Task 3: Example Names

**Files:**
- Modify: `examples/train_*.py`

- [ ] Rename `s_dim` to `state_dim`.
- [ ] Rename `a_dim` to `action_dim`.
- [ ] Rename `a_low` and `a_high` to `action_low` and `action_high`.
- [ ] Rename `bb` and `tbb` to `backbone` and `target_backbone`.
- [ ] Rename `q`, `tq`, and `q_net` to `q_network` and `target_q_network`.
- [ ] Rename example helper functions such as `build_cnn_q_net` to `build_cnn_q_network`.

### Task 4: Docs and Tests

**Files:**
- Modify: `README.md` only if examples mention renamed helpers.
- Test: all tests.

- [ ] Run `.venv/bin/python -m pytest -v`.
- [ ] Fix any import or naming errors.
