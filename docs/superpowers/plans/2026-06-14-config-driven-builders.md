# Config-Driven Builders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-driven `build_trainer` API that assembles experiments from dictionaries.

**Architecture:** Add a focused builder module under `modular_rl.training`, expand `Registry` into a fuller component catalog, and keep existing quick factories as compatibility wrappers. Tests verify construction behavior without running long RL training loops.

**Tech Stack:** Python 3.8+, PyTorch, Gymnasium, pytest.

---

### Task 1: Builder Tests

**Files:**
- Create: `tests/test_config_builders.py`

- [x] **Step 1: Write failing tests**

Added tests for minimal DQN, recurrent DQN, SAC/PPO/TD3 construction, unknown algorithm errors, and registry introspection.

- [x] **Step 2: Run tests and verify they fail**

Initial test execution failed before implementation because `build_trainer` was not exported.

### Task 2: Registry Expansion

**Files:**
- Modify: `modular_rl/training/registry.py`

- [x] **Step 1: Add registries for configs, buffers, envs, loggers, optimizers**

Added registration, build, and list methods for component groups while preserving existing backbone/head/agent helpers.

- [x] **Step 2: Run focused tests**

Registry introspection tests pass.

### Task 3: Config Builder

**Files:**
- Create: `modular_rl/training/builders.py`
- Modify: `modular_rl/training/__init__.py`

- [x] **Step 1: Implement `ExperimentBuilder` and `build_trainer`**

Implemented DQN, SAC, PPO, and TD3 construction using the current quick factory behavior as the compatibility baseline.

- [x] **Step 2: Export builder API**

Exported `ExperimentBuilder`, `BuildContext`, and `build_trainer` from `training.__init__`.

- [x] **Step 3: Run builder tests**

`tests/test_config_builders.py` passes.

### Task 4: Compatibility and Docs

**Files:**
- Modify: `modular_rl/training/factory.py`
- Modify: `README.md`

- [x] **Step 1: Route quick factories through `build_trainer`**

Kept function signatures stable and delegated construction to shared config specs.

- [x] **Step 2: Document config-driven usage**

Added README usage for `build_trainer` and registry extension points.

- [x] **Step 3: Run verification**

`.venv/bin/python -m pytest -v` passes.
