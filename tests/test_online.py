"""在线动态标定的质量门控、平滑与突变重锚定测试。"""
from __future__ import annotations

import numpy as np

from dst_calib.online import (
    OnlineExtrinsicGate,
    interpolate_se3,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rotation_distance_deg,
    validate_se3,
)


def _T_z(deg: float, xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = xyz
    return T


def test_quaternion_roundtrip_including_pi():
    for deg in (0.0, 45.0, 179.999, 180.0):
        R = _T_z(deg)[:3, :3]
        q = matrix_to_quaternion_xyzw(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert rotation_distance_deg(quaternion_xyzw_to_matrix(q), R) < 1e-5


def test_interpolate_se3_translation_and_rotation():
    T = interpolate_se3(np.eye(4), _T_z(2.0, (1.0, 0.0, 0.0)), 0.25)
    assert np.allclose(T[:3, 3], [0.25, 0.0, 0.0])
    assert np.isclose(
        rotation_distance_deg(np.eye(3), T[:3, :3]), 0.5, atol=2e-5)


def test_gate_accepts_first_then_smooths_small_update():
    gate = OnlineExtrinsicGate(
        max_final_cd=1.0,
        max_rotation_jump_deg=5.0,
        max_translation_jump_m=0.3,
        smoothing_alpha=0.25,
    )
    first = gate.update(np.eye(4), 0.02)
    assert first.accepted
    second = gate.update(_T_z(2.0, (0.20, 0.0, 0.0)), 0.03)
    assert second.accepted and not second.reanchored
    assert np.allclose(second.transform[:3, 3], [0.05, 0.0, 0.0])
    assert np.isclose(
        rotation_distance_deg(np.eye(3), second.transform[:3, :3]),
        0.5,
        atol=2e-5,
    )


def test_gate_rejects_bad_quality_without_changing_current():
    gate = OnlineExtrinsicGate(max_final_cd=0.1)
    gate.update(np.eye(4), 0.02)
    before = gate.current
    result = gate.update(_T_z(1.0), 0.2)
    assert not result.accepted
    assert "final_cd" in result.reason
    assert np.allclose(gate.current, before)


def test_large_jump_requires_consistent_confirmation_then_reanchors():
    gate = OnlineExtrinsicGate(
        max_final_cd=1.0,
        max_rotation_jump_deg=5.0,
        max_translation_jump_m=0.30,
        smoothing_alpha=0.5,
        reanchor_confirmations=2,
        confirmation_rotation_deg=1.0,
        confirmation_translation_m=0.05,
    )
    gate.update(np.eye(4), 0.02)
    first_jump = gate.update(_T_z(9.0, (0.40, 0.0, 0.0)), 0.03)
    assert not first_jump.accepted
    assert gate.pending_count == 1
    confirmed = gate.update(_T_z(9.2, (0.41, 0.0, 0.0)), 0.03)
    assert confirmed.accepted and confirmed.reanchored
    assert gate.pending_count == 0
    assert rotation_distance_deg(
        confirmed.transform[:3, :3], _T_z(9.1)[:3, :3]) < 1e-5
    assert np.allclose(confirmed.transform[:3, 3], [0.405, 0.0, 0.0])


def test_inconsistent_large_jumps_restart_confirmation():
    gate = OnlineExtrinsicGate(
        max_final_cd=1.0,
        max_rotation_jump_deg=2.0,
        reanchor_confirmations=2,
        confirmation_rotation_deg=0.5,
    )
    gate.update(np.eye(4), 0.01)
    gate.update(_T_z(8.0), 0.01)
    result = gate.update(_T_z(-8.0), 0.01)
    assert not result.accepted
    assert gate.pending_count == 1


def test_validate_se3_rejects_reflection_and_nan():
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    assert not validate_se3(reflection)[0]
    invalid = np.eye(4)
    invalid[0, 3] = np.nan
    assert not validate_se3(invalid)[0]
