"""反求（goal seek）测试：二分求根、单位换算、错误中止与不返回伪解。"""
import math

import pytest

from app.engine import analyze, goal_seek


def solve_ok(expr, variables, variable, unit, lower, upper, target, **kw):
    res = goal_seek(expr, variables, variable, unit, lower, upper, target, **kw)
    assert res["ok"], res["errors"]
    assert res["result"] is not None
    return res["result"]


def solve_fail(expr, variables, variable, unit, lower, upper, target, **kw):
    res = goal_seek(expr, variables, variable, unit, lower, upper, target, **kw)
    assert not res["ok"]
    assert res["result"] is None
    assert res["errors"]
    return res["errors"]


# ---------------- 基本二分求根 ----------------

def test_solve_velocity_for_kinetic_energy():
    # E = 0.5 m v^2 = 100 J，m = 2 kg → v = 10 m/s
    r = solve_ok("0.5*mass*v^2", {"mass": "2 kg", "v": "1 m/s"},
                 "v", "m/s", 0.0, 20.0, "100 J")
    assert r["variable"] == "v"
    assert r["variable_si_value"] == pytest.approx(10.0, abs=1e-7)
    assert r["value"] == pytest.approx(10.0, abs=1e-7)
    assert r["unit"] == "m/s"
    assert r["variable_si_unit"] == "m·s^-1"
    assert r["achieved_si_value"] == pytest.approx(100.0, abs=1e-7)
    assert abs(r["residual_si"]) <= r["tolerance_si"]
    assert r["iterations"] <= 80
    assert r["method"] == "bisection"


def test_solve_target_in_prefixed_unit():
    # 目标用 kJ 给出：0.5*2*v^2 = 0.1 kJ = 100 J → v = 10 m/s
    r = solve_ok("0.5*mass*v^2", {"mass": "2 kg", "v": "1 m/s"},
                 "v", "m/s", 0.0, 20.0, "0.1 kJ")
    assert r["target_si_value"] == pytest.approx(100.0)
    assert r["variable_si_value"] == pytest.approx(10.0, abs=1e-7)


def test_solve_variable_unit_prefix_conversion():
    # F = mass*g0；P = F/A，求 A 使 P = 100 kPa；A 以 cm² 给出
    variables = {"mass": "80 kg", "F": "mass * g0", "A": "1 m^2"}
    r = solve_ok("F/A", variables, "A", "cm^2", 10.0, 1000.0, "100 kPa")
    a_si = 80 * 9.80665 / 100_000
    assert r["variable_si_value"] == pytest.approx(a_si, rel=1e-7)
    assert r["value"] == pytest.approx(a_si * 1e4, rel=1e-7)  # m² → cm²
    # 残差只需满足规定容差 1e-8·max(1, |目标 SI|) = 1e-3 Pa
    assert abs(r["residual_si"]) <= r["tolerance_si"]


def test_solve_dimensionless_variable():
    r = solve_ok("x^2", {"x": "1"}, "x", "rad", 0.0, 4.0, "9")
    assert r["variable_si_value"] == pytest.approx(3.0, abs=1e-7)
    assert r["achieved_dimension_str"] == "无量纲"


def test_solve_negative_root():
    r = solve_ok("x", {"x": "1 m"}, "x", "m", -10.0, 0.0, "-5.5 m")
    assert r["variable_si_value"] == pytest.approx(-5.5)
    assert r["value"] == pytest.approx(-5.5)


# ---------------- 端点达标：直接返回 ----------------

def test_endpoint_already_at_target_returns_immediately():
    r = solve_ok("0.5*mass*v^2", {"mass": "2 kg", "v": "1 m/s"},
                 "v", "m/s", 0.0, 20.0, "400 J")
    assert r["iterations"] == 0
    assert r["variable_si_value"] == pytest.approx(20.0)
    assert r["residual_si"] == pytest.approx(0.0)


def test_lower_endpoint_at_target():
    r = solve_ok("x", {"x": "1 m"}, "x", "m", 3.0, 9.0, "3 m")
    assert r["iterations"] == 0
    assert r["variable_si_value"] == pytest.approx(3.0)


# ---------------- 参数校验错误：不返回解 ----------------

def test_unknown_variable_rejected():
    errs = solve_fail("x", {"x": "1 m"}, "y", "m", 0.0, 2.0, "1 m")
    assert errs[0]["kind"] == "solve_var_unknown"
    assert errs[0]["source"] == "solve:variable"


def test_indirect_variable_rejected():
    errs = solve_fail("F", {"mass": "2 kg", "F": "mass*g0"},
                      "F", "N", 0.0, 100.0, "50 N")
    assert errs[0]["kind"] == "solve_var_indirect"
    assert errs[0]["source"] == "solve:variable"


def test_variable_unit_dimension_mismatch_rejected():
    errs = solve_fail("0.5*mass*v^2", {"mass": "2 kg", "v": "1 m/s"},
                      "v", "s", 0.0, 20.0, "100 J")
    assert errs[0]["kind"] == "solve_unit_dim"
    assert errs[0]["source"] == "solve:unit"


def test_unknown_variable_unit_rejected():
    errs = solve_fail("x", {"x": "1 m"}, "x", "furlong", 0.0, 2.0, "1 m")
    assert errs[0]["kind"] == "undefined"
    assert errs[0]["source"] == "solve:unit"


def test_target_dimension_mismatch_rejected():
    errs = solve_fail("0.5*mass*v^2", {"mass": "2 kg", "v": "1 m/s"},
                      "v", "m/s", 0.0, 20.0, "100 W")
    assert errs[0]["kind"] == "solve_target_dim"
    assert errs[0]["source"] == "solve:target"


def test_bounds_must_be_finite_and_ordered():
    errs = solve_fail("x", {"x": "1 m"}, "x", "m", float("nan"), 2.0, "1 m")
    assert any(e["kind"] == "solve_bound" for e in errs)
    errs = solve_fail("x", {"x": "1 m"}, "x", "m", 0.0, float("inf"), "1 m")
    assert any(e["kind"] == "solve_bound" for e in errs)
    errs = solve_fail("x", {"x": "1 m"}, "x", "m", 2.0, 2.0, "1 m")
    assert errs[0]["kind"] == "solve_bound_order"
    assert errs[0]["source"] == "solve:bounds"


def test_missing_unit_and_target_rejected():
    res = goal_seek("x", {"x": "1 m"}, "x", "", 0.0, 2.0, "")
    assert not res["ok"]
    kinds = {e["kind"] for e in res["errors"]}
    assert "solve_unit" in kinds and "solve_target" in kinds


# ---------------- 区间不包根 / 求值失败 / 超限 ----------------

def test_interval_without_sign_change_rejected():
    errs = solve_fail("0.5*mass*v^2", {"mass": "2 kg", "v": "1 m/s"},
                      "v", "m/s", 20.0, 30.0, "100 J")
    assert errs[0]["kind"] == "solve_no_bracket"
    assert errs[0]["source"] == "solve:bounds"
    assert errs[0]["residual_lower_si"] > 0
    assert errs[0]["residual_upper_si"] > 0


def test_midpoint_evaluation_failure_aborts_without_solution():
    # 1 m² / x 在 x=0 处除零；区间 [-2, 2] 对称，第一个中点恰为 0
    errs = solve_fail("1 m^2 / x", {"x": "1 m"}, "x", "m",
                      -2.0, 2.0, "0 m")
    assert errs[0]["kind"] == "value"
    assert errs[0]["solve_var"] == "x"
    assert errs[0]["where"] == "内"


def test_endpoint_evaluation_failure_aborts():
    # 下界为负，sqrt 在负端无定义
    errs = solve_fail("sqrt(x)", {"x": "1 m^2"}, "x", "m^2",
                      -1.0, 16.0, "2 m")
    assert errs[0]["kind"] == "value"
    assert "sqrt" in errs[0]["message"]
    assert errs[0]["where"] == "下界"


def test_exceeding_max_iterations_reports_no_convergence():
    # 区间 [0,1]、目标 1/3：2 次二分远达不到 1e-8 相对容差
    errs = solve_fail("x", {"x": "1 m"}, "x", "m",
                      0.0, 1.0, "0.3333333333333333 m", max_iter=2)
    assert errs[0]["kind"] == "solve_no_convergence"
    assert errs[0]["iterations"] == 2


def test_float_stagnation_reports_no_convergence():
    # tan 在 π/2 两侧相邻双精度数之间：两端残差异号（±∞ 方向），
    # 但首次中点即舍回端点，区间无法继续收缩且残差远超容差
    lo = math.nextafter(math.pi / 2, 0.0)
    hi = math.nextafter(math.pi / 2, math.inf)
    assert hi > lo
    errs = solve_fail("tan(x)", {"x": "1 rad"}, "x", "rad",
                      lo, hi, "0", max_iter=80)
    assert errs[0]["kind"] == "solve_no_convergence"


# ---------------- 与现有校验一致的行为 ----------------

def test_baseline_validation_still_runs():
    # 既有量纲错误必须原样报出，不进入反求
    errs = solve_fail("1 m + 1 s", {}, "x", "m", 0.0, 2.0, "1 m")
    assert errs[0]["kind"] == "dim_mismatch"
    assert errs[0]["source"] == "expression"


def test_shadow_warning_preserved():
    # 与内置单位同名但定义合法的直接输入
    res = goal_seek("2*l", {"l": "1 m"}, "l", "m", 0.0, 2.0, "1.5 m")
    assert res["ok"], res["errors"]
    assert any(w["kind"] == "shadow_unit" and w["name"] == "l"
               for w in res["warnings"])


def test_analyze_remains_backward_compatible():
    res = analyze("0.5*mass*v^2", {"mass": "2 kg", "v": "10 m/s"}, "kJ")
    assert res["ok"]
    assert res["result"]["si_value"] == pytest.approx(100.0)
    assert res["result"]["target"]["value"] == pytest.approx(0.1)
