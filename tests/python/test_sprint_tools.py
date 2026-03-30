"""Wave 11 validation tests — Sprint result tooling.

Tests required by DREX_UNIFIED_PLAN.md § Sprint Checklist:
  1. ParseLogExtractsVal   : parse_log returns correct [{step, val_ppl}] from synthetic log
  2. ParseLogNoValLines    : parse_log returns [] when no [val] lines present
  3. ParseLogEmptyInput    : parse_log returns [] for empty-content file
  4. ExtractAtStepFound    : extract_at_step returns correct ppl when step present
  5. ExtractAtStepMissing  : extract_at_step returns None when step not in records
  6. CheckGatePassBelow    : check_gate returns True when val_ppl < gate
  7. CheckGateFailAbove    : check_gate returns False when val_ppl > gate
  8. CheckGateExactlyAt    : check_gate returns True when val_ppl == gate (inclusive)
"""

import os
import tempfile

import pytest

from scripts.record_sprint_results import (
    check_gate,
    extract_at_step,
    parse_log,
)

# ---------------------------------------------------------------------------
# Synthetic log content
# ---------------------------------------------------------------------------

_SYNTHETIC_LOG = """\
  [train] step    500  loss 0.7200
  [val] step    500  val_loss 0.6359  val_ppl    1.89
  [train] step   1000  loss 0.4100
  [val] step   1000  val_loss 0.0291  val_ppl    1.03
  [val] step   5000  val_loss 0.0210  val_ppl    1.02
  [val] step  10000  val_loss 0.0198  val_ppl    1.01
"""

_NO_VAL_LOG = """\
  [train] step    500  loss 0.7200
  [train] step   1000  loss 0.4100
"""


def _write_tmp(content: str) -> str:
    """Write content to a NamedTemporaryFile. Caller must os.unlink the returned path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests 1–8
# ---------------------------------------------------------------------------

def test_parse_log_extracts_val():
    """1. parse_log returns correct [{step, val_ppl}] sorted by step."""
    path = _write_tmp(_SYNTHETIC_LOG)
    try:
        records = parse_log(path)
    finally:
        os.unlink(path)

    assert len(records) == 4
    assert records[0] == {"step": 500, "val_ppl": 1.89}
    assert records[1] == {"step": 1000, "val_ppl": 1.03}
    assert records[2] == {"step": 5000, "val_ppl": 1.02}
    assert records[3] == {"step": 10000, "val_ppl": 1.01}


def test_parse_log_no_val_lines():
    """2. parse_log returns [] when no [val] lines present."""
    path = _write_tmp(_NO_VAL_LOG)
    try:
        records = parse_log(path)
    finally:
        os.unlink(path)

    assert records == []


def test_parse_log_empty_input():
    """3. parse_log returns [] for empty-content file."""
    path = _write_tmp("")
    try:
        records = parse_log(path)
    finally:
        os.unlink(path)

    assert records == []


def test_extract_at_step_found():
    """4. extract_at_step returns correct ppl when step present."""
    records = [
        {"step": 500, "val_ppl": 1.89},
        {"step": 1000, "val_ppl": 1.03},
        {"step": 10000, "val_ppl": 1.01},
    ]
    assert extract_at_step(records, 1000) == pytest.approx(1.03)


def test_extract_at_step_missing():
    """5. extract_at_step returns None when step not in records."""
    records = [{"step": 500, "val_ppl": 1.89}, {"step": 10000, "val_ppl": 1.01}]
    assert extract_at_step(records, 9999) is None


def test_check_gate_pass_below():
    """6. check_gate returns True when val_ppl < gate."""
    assert check_gate(1.08, 1.32) is True


def test_check_gate_fail_above():
    """7. check_gate returns False when val_ppl > gate."""
    assert check_gate(1.45, 1.32) is False


def test_check_gate_exactly_at():
    """8. check_gate returns True when val_ppl == gate (inclusive)."""
    assert check_gate(1.32, 1.32) is True
