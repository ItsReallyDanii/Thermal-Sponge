"""
test_bito_smoke.py — Minimal smoke tests for the canonical BITO pipeline.

These tests verify that the core BITO components are importable, internally
consistent, and produce sane outputs on trivial inputs. They are fast (< 5s
total) and require no external data files.

Canonical files under test:
  src/constants.py
  src/heat_simulation_transient.py
  src/pid_controller.py
  src/flytrap_gate.py
  src/control_metrics.py
  src/run_thermal_orchestration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for src.* imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_constants_values():
    """Constants must match the values that the BITO physics depend on."""
    from src.constants import (
        VOID_THRESHOLD,
        K_SOLID,
        K_VOID,
        T_SLA_DEFAULT,
        DT_DEFAULT,
        N_STEPS_DEFAULT,
        K_VOID_PASSIVE,
        K_VOID_ACTIVE,
        GATE_N_TRIGGER,
        GATE_T_WINDOW,
        GATE_T_REFRACTORY,
    )

    # Threshold must be above 0.5 (original bug was 0.5); canonical value is 0.60
    assert VOID_THRESHOLD == 0.60, f"VOID_THRESHOLD changed from canonical 0.60: {VOID_THRESHOLD}"

    # CFL stability: dt <= 1 / (4 * K_VOID_ACTIVE)
    cfl_limit = 1.0 / (4.0 * K_VOID_ACTIVE)
    assert DT_DEFAULT <= cfl_limit, (
        f"DT_DEFAULT={DT_DEFAULT} violates CFL limit {cfl_limit} for K_VOID_ACTIVE={K_VOID_ACTIVE}"
    )

    # Passive void must be much less than solid (stagnant cooling off)
    assert K_VOID_PASSIVE < K_SOLID, "K_VOID_PASSIVE must be less than K_SOLID"

    # Active void must be greater than solid (forced convection >> conduction)
    assert K_VOID_ACTIVE > K_SOLID, "K_VOID_ACTIVE must exceed K_SOLID for forced-convection effect"

    # SLA in (0, 1) normalized range
    assert 0.0 < T_SLA_DEFAULT < 1.0, f"T_SLA_DEFAULT must be in (0, 1): {T_SLA_DEFAULT}"

    # Gate parameters must be positive integers
    assert GATE_N_TRIGGER >= 1
    assert GATE_T_WINDOW >= 1
    assert GATE_T_REFRACTORY >= 1


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------

def test_pid_zero_setpoint_zero_measured():
    """PID with zero error should output near zero (no integral wind-up from start)."""
    from src.pid_controller import PIDController
    pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, dt=0.1, setpoint=0.0)
    u = pid.step(measured=0.0)
    assert u == 0.0, f"Expected 0.0 from zero-error PID, got {u}"


def test_pid_positive_error_drives_output():
    """Positive error (measured < setpoint) should drive output positive."""
    from src.pid_controller import PIDController
    pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, dt=0.1, setpoint=1.0)
    u = pid.step(measured=0.0)  # error = 1.0 -> u = Kp * error = 2.0, clamped to 1.0
    assert u == 1.0, f"Expected clamped output 1.0, got {u}"


def test_pid_anti_windup():
    """Integral should not accumulate when saturated."""
    from src.pid_controller import PIDController
    pid = PIDController(Kp=10.0, Ki=10.0, Kd=0.0, dt=0.1, setpoint=1.0)
    for _ in range(100):
        u = pid.step(measured=0.0)  # always saturated high
    # Integral should not have wound up to infinity
    assert pid._integral < 1e6, "Integral wound up despite anti-windup"
    assert u == 1.0, "Saturated output should be clamped to u_max"


def test_pid_reset():
    """Reset clears internal state."""
    from src.pid_controller import PIDController
    pid = PIDController(Kp=1.0, Ki=1.0, Kd=0.0, dt=0.1, setpoint=1.0)
    for _ in range(10):
        pid.step(measured=0.0)
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error == 0.0


# ---------------------------------------------------------------------------
# FlyTrapGate
# ---------------------------------------------------------------------------

def test_flytrap_gate_opens_after_n_triggers():
    """Gate should open after N_trigger events within T_window."""
    from src.flytrap_gate import FlyTrapGate
    gate = FlyTrapGate(N_trigger=3, T_window=10, T_refractory=5)

    # 3 triggers in steps 0, 1, 2 → gate primes at step 2, opens at step 3
    assert gate.update(True, 0) is False  # IDLE, 1 event
    assert gate.update(True, 1) is False  # IDLE, 2 events
    assert gate.update(True, 2) is False  # PRIMED (transition detected, but returns False for IDLE)
    open_result = gate.update(True, 3)    # OPEN
    assert open_result is True, "Gate should be open after N triggers"


def test_flytrap_gate_refractory():
    """Gate should refuse to reopen during refractory period."""
    from src.flytrap_gate import FlyTrapGate
    gate = FlyTrapGate(N_trigger=2, T_window=10, T_refractory=5)

    # Force open
    gate.update(True, 0)
    gate.update(True, 1)
    gate.update(True, 2)  # OPEN

    # Close by sending no trigger
    gate.update(False, 3)  # REFRACTORY starts

    # During refractory, triggers should not reopen gate
    for t in range(4, 8):
        result = gate.update(True, t)
        assert result is False, f"Gate opened during refractory at t={t}"


def test_flytrap_gate_reset():
    """Reset returns gate to IDLE state."""
    from src.flytrap_gate import FlyTrapGate
    gate = FlyTrapGate(N_trigger=2, T_window=10, T_refractory=5)
    gate.update(True, 0)
    gate.update(True, 1)
    gate.update(True, 2)  # now OPEN
    gate.reset()
    assert gate.state == "IDLE"
    assert gate.transition_count == 0


# ---------------------------------------------------------------------------
# Control metrics
# ---------------------------------------------------------------------------

def test_control_metrics_all_compliant():
    """When T_max always below SLA, violations=0 and settling_time=0."""
    from src.control_metrics import compute_control_metrics
    n = 50
    u_history = np.ones(n, dtype=np.float32)
    T_max_history = np.full(n, 0.5, dtype=np.float32)
    T_SLA = 0.85

    m = compute_control_metrics(u_history, T_max_history, T_SLA, dt=0.1)
    assert m.sla_violations == 0
    assert m.sla_max_exceedance == 0.0
    assert m.settling_time == 0


def test_control_metrics_kWh_ctrl():
    """kWh_ctrl = sum(|u|) * dt."""
    from src.control_metrics import compute_control_metrics
    n = 10
    u_history = np.ones(n, dtype=np.float32)
    T_max_history = np.zeros(n, dtype=np.float32)
    dt = 0.1
    m = compute_control_metrics(u_history, T_max_history, T_SLA=0.85, dt=dt)
    expected_kWh = n * dt  # sum(|1|) * 0.1 = 1.0
    assert abs(m.kWh_ctrl - expected_kWh) < 1e-6


def test_control_metrics_chatter_count():
    """Chatter = number of 0→1 transitions in binarized u."""
    from src.control_metrics import compute_control_metrics
    # u pattern: 0 1 0 1 0 → 2 transitions (0→1)
    u_history = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    T_max_history = np.zeros(5, dtype=np.float32)
    m = compute_control_metrics(u_history, T_max_history, T_SLA=0.85, dt=0.1)
    assert m.chatter_count == 2, f"Expected 2 chatter transitions, got {m.chatter_count}"


# ---------------------------------------------------------------------------
# Transient heat solver (short smoke run)
# ---------------------------------------------------------------------------

def test_transient_solver_runs_and_heats():
    """Transient solver should run without error and produce nonzero temperatures."""
    from src.heat_simulation_transient import solve_transient_heat

    # Small grid, short run
    shape = (32, 32)
    img = np.ones(shape, dtype=np.float32)  # all void (bright)
    gate_always_on = lambda step, T_max: 1.0

    result = solve_transient_heat(
        img_arr=img,
        gate_signal=gate_always_on,
        dt=0.1,
        n_steps=20,
        q_chip=0.005,
    )

    assert result.T_final.shape == shape
    assert result.T_max_history.shape == (20,)
    assert result.u_history.shape == (20,)
    # Temperature should have risen from 0
    assert result.T_max_history[-1] > 0.0, "Chip temperature should rise with heat source"
    # Gate was always on, so all u values should be 1.0
    assert np.all(result.u_history == 1.0)


def test_transient_solver_cfl_check():
    """Solver should raise ValueError if dt violates CFL stability."""
    from src.heat_simulation_transient import solve_transient_heat

    shape = (16, 16)
    img = np.ones(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="CFL"):
        solve_transient_heat(
            img_arr=img,
            gate_signal=lambda s, T: 1.0,
            dt=0.5,  # far exceeds CFL limit of 0.125
            n_steps=5,
        )


def test_transient_solver_gate_off_heats_more():
    """With gate off, temperature should rise higher than with gate on."""
    from src.heat_simulation_transient import solve_transient_heat

    shape = (32, 32)
    img = np.ones(shape, dtype=np.float32)

    result_on = solve_transient_heat(
        img_arr=img,
        gate_signal=lambda s, T: 1.0,
        dt=0.1,
        n_steps=50,
        q_chip=0.005,
    )
    result_off = solve_transient_heat(
        img_arr=img,
        gate_signal=lambda s, T: 0.0,
        dt=0.1,
        n_steps=50,
        q_chip=0.005,
    )

    T_max_on = result_on.T_max_history[-1]
    T_max_off = result_off.T_max_history[-1]
    assert T_max_off > T_max_on, (
        f"Gate-off should yield higher temperature (got off={T_max_off:.4f}, on={T_max_on:.4f})"
    )


# ---------------------------------------------------------------------------
# Orchestration: morphology generators (no heavy simulation)
# ---------------------------------------------------------------------------

def test_generate_morphology_shapes():
    """All named morphologies should return correct shapes and value ranges."""
    from src.run_thermal_orchestration import generate_morphology

    for morph in ["Fins_20", "Random_0.4", "Xylem"]:
        img = generate_morphology(morph, seed=42)
        assert img.shape == (256, 256), f"{morph}: wrong shape {img.shape}"
        assert img.min() >= 0.0 and img.max() <= 1.0, f"{morph}: values out of [0, 1]"
        assert img.dtype == np.float32, f"{morph}: wrong dtype {img.dtype}"


def test_generate_morphology_deterministic():
    """Same seed should produce identical morphologies."""
    from src.run_thermal_orchestration import generate_morphology

    for morph in ["Fins_20", "Random_0.4", "Xylem"]:
        img1 = generate_morphology(morph, seed=7)
        img2 = generate_morphology(morph, seed=7)
        assert np.array_equal(img1, img2), f"{morph}: not deterministic across calls"


def test_generate_morphology_seed_differs():
    """Different seeds should produce different Random and Xylem morphologies."""
    from src.run_thermal_orchestration import generate_morphology

    for morph in ["Random_0.4", "Xylem"]:
        img1 = generate_morphology(morph, seed=1)
        img2 = generate_morphology(morph, seed=99)
        assert not np.array_equal(img1, img2), f"{morph}: seed has no effect"
