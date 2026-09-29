"""量纲推导引擎：词法分析 → 递归下降解析 → 量纲/数值求值。

支持：
  - 四则运算、幂运算（^ 或 **，右结合）、括号、一元正负号
  - 隐式乘法：``2 m``、``2(1+1)``、``kg m/s^2``、``2π r``
  - 数字后跟单位：``9.8 m/s^2``、``1.5e-3 mol``
  - 上标 Unicode：``m²``、``x⁻¹``
  - 全角/符号运算符：× ÷ · −
  - 函数调用：sqrt / abs / sin / cos / tan / asin / acos / atan / exp / ln / log / log2
  - 变量引用（变量值本身也是带单位的表达式，可相互引用，检测循环）

所有错误都携带 (start, end) 片段位置，指回原始表达式字符串。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from .units import (
    CONSTANTS, DIMLESS, DIM_NAMES, FUNCTIONS, UNITS,
    derived_matches, dim_str, format_fraction, si_unit_str,
)

# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------

@dataclass
class AnalysisError:
    message: str
    start: int
    end: int
    kind: str = "error"
    source: str = "expression"  # expression / target / var:<name> / uncert:<name>
    extra: dict | None = None   # 附加字段，如扰动来源 {"perturbed": name, "direction": "+" | "-"}

    def to_dict(self) -> dict:
        d = {
            "message": self.message,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "source": self.source,
        }
        if self.extra:
            d.update(self.extra)
        return d


class EngineError(Exception):
    """携带片段位置的引擎内部异常。"""

    def __init__(self, message: str, start: int, end: int, kind: str = "error"):
        super().__init__(message)
        self.error = AnalysisError(message, start, end, kind)


# ---------------------------------------------------------------------------
# 词法分析
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str          # num / ident / op / eof
    text: str
    start: int
    end: int
    value: float = 0.0


_NUM_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-",
}

# 单字符运算符（含常见全角/符号别名，长度均为 1，不影响片段定位）
_CHAR_OPS = {
    "+": "+", "-": "-", "−": "-",
    "*": "*", "×": "*", "·": "*", "⋅": "*",
    "/": "/", "÷": "/",
    "^": "^", "(": "(", ")": ")", ",": ",",
}

_IDENT_EXTRA_START = "_°%"   # isalpha() 之外的标识符起始字符
_IDENT_EXTRA_CONT = "_"


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        # 注意：'²' 等上标字符的 isdigit() 为真，数字判断必须限定 ASCII
        if ch in "0123456789" or (ch == "." and i + 1 < n and src[i + 1] in "0123456789"):
            m = _NUM_RE.match(src, i)
            assert m is not None
            text = m.group()
            tokens.append(Token("num", text, i, m.end(), float(text)))
            i = m.end()
            continue
        if ch in _SUPERSCRIPTS:
            j = i
            digits = ""
            while j < n and src[j] in _SUPERSCRIPTS:
                digits += _SUPERSCRIPTS[src[j]]
                j += 1
            # 上标序列 → 幂运算：^ <digits>（^ 为零宽 token，指向上标起点）
            tokens.append(Token("op", "^", i, i))
            tokens.append(Token("num", digits, i, j, float(digits)))
            i = j
            continue
        if ch.isalpha() or ch in _IDENT_EXTRA_START:
            j = i + 1
            # 注意：'²' 等上标字符的 isalnum() 为真，必须显式排除
            while j < n and src[j] not in _SUPERSCRIPTS and (
                src[j].isalnum() or src[j] in _IDENT_EXTRA_CONT
            ):
                j += 1
            tokens.append(Token("ident", src[i:j], i, j))
            i = j
            continue
        if ch == "*" and i + 1 < n and src[i + 1] == "*":
            tokens.append(Token("op", "^", i, i + 2))
            i += 2
            continue
        if ch in _CHAR_OPS:
            tokens.append(Token("op", _CHAR_OPS[ch], i, i + 1))
            i += 1
            continue
        raise EngineError(f"无法识别的字符「{ch}」", i, i + 1, "syntax")
    tokens.append(Token("eof", "", n, n))
    return tokens


# ---------------------------------------------------------------------------
# 语法分析（递归下降）
# ---------------------------------------------------------------------------

@dataclass
class Node:
    kind: str                       # num / name / neg / bin / pow / call / group
    start: int
    end: int
    value: float = 0.0
    name: str = ""
    op: str = ""
    left: "Node | None" = None
    right: "Node | None" = None
    args: list = field(default_factory=list)


class Parser:
    """文法（优先级由低到高）：
        add  := mul  (('+'|'-') mul)*
        mul  := unary (('*'|'/'|隐式) unary)*
        unary:= ('-'|'+') unary | pow
        pow  := primary ('^' unary)?          （右结合；指数允许负号）
        primary := NUMBER | IDENT | IDENT '(' args ')' | '(' add ')'
    隐式乘法：primary 之后紧跟 NUMBER / IDENT / '(' 时视为相乘。
    """

    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def advance(self) -> Token:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect_op(self, op: str) -> Token:
        tok = self.peek()
        if tok.kind != "op" or tok.text != op:
            raise EngineError(f"此处应为「{op}」", tok.start, max(tok.end, tok.start + 1), "syntax")
        return self.advance()

    def parse(self) -> Node:
        if self.peek().kind == "eof":
            raise EngineError("表达式为空", 0, 0, "syntax")
        node = self.parse_add()
        tok = self.peek()
        if tok.kind != "eof":
            raise EngineError(f"此处有多余的内容「{tok.text}」", tok.start, max(tok.end, tok.start + 1), "syntax")
        return node

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self.peek().kind == "op" and self.peek().text in ("+", "-"):
            op = self.advance()
            right = self.parse_mul()
            left = Node("bin", left.start, right.end, op=op.text, left=left, right=right)
        return left

    def parse_mul(self) -> Node:
        left = self.parse_unary()
        while True:
            tok = self.peek()
            if tok.kind == "op" and tok.text in ("*", "/"):
                self.advance()
                right = self.parse_unary()
                left = Node("bin", left.start, right.end, op=tok.text, left=left, right=right)
            elif tok.kind in ("num", "ident") or (tok.kind == "op" and tok.text == "("):
                right = self.parse_unary()  # 隐式乘法
                left = Node("bin", left.start, right.end, op="*", left=left, right=right)
            else:
                return left

    def parse_unary(self) -> Node:
        tok = self.peek()
        if tok.kind == "op" and tok.text == "-":
            self.advance()
            operand = self.parse_unary()
            return Node("neg", tok.start, operand.end, left=operand)
        if tok.kind == "op" and tok.text == "+":
            self.advance()
            return self.parse_unary()
        return self.parse_pow()

    def parse_pow(self) -> Node:
        base = self.parse_primary()
        tok = self.peek()
        if tok.kind == "op" and tok.text == "^":
            self.advance()
            exponent = self.parse_unary()
            return Node("pow", base.start, exponent.end, left=base, right=exponent)
        return base

    def parse_primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "num":
            self.advance()
            return Node("num", tok.start, tok.end, value=tok.value)
        if tok.kind == "ident":
            self.advance()
            name = tok.text
            nxt = self.peek()
            if name in FUNCTIONS and nxt.kind == "op" and nxt.text == "(":
                self.advance()
                args: list[Node] = []
                if not (self.peek().kind == "op" and self.peek().text == ")"):
                    args.append(self.parse_add())
                    while self.peek().kind == "op" and self.peek().text == ",":
                        self.advance()
                        args.append(self.parse_add())
                rpar = self.expect_op(")")
                return Node("call", tok.start, rpar.end, name=name, args=args)
            return Node("name", tok.start, tok.end, name=name)
        if tok.kind == "op" and tok.text == "(":
            self.advance()
            inner = self.parse_add()
            rpar = self.expect_op(")")
            return Node("group", tok.start, rpar.end, left=inner)
        raise EngineError("此处应为数字、变量、单位或括号", tok.start, max(tok.end, tok.start + 1), "syntax")


# ---------------------------------------------------------------------------
# 求值（数值 + 量纲递归推导）
# ---------------------------------------------------------------------------

@dataclass
class Quantity:
    value: float      # 以 SI 基本单位计的数值
    dim: tuple        # 量纲向量


def _dim_add(a: tuple, b: tuple) -> tuple:
    return tuple(x + y for x, y in zip(a, b))


def _dim_sub(a: tuple, b: tuple) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


def _dim_scale(d: tuple, k: Fraction) -> tuple:
    return tuple(x * k for x in d)


class Evaluator:
    """对 AST 递归求值。resolve(name, node) 由外层注入，用于解析变量/单位/常量。"""

    def __init__(self, resolve):
        self.resolve = resolve

    def eval(self, node: Node) -> Quantity:
        kind = node.kind
        if kind == "num":
            return Quantity(node.value, DIMLESS)
        if kind == "group":
            return self.eval(node.left)
        if kind == "name":
            return self.resolve(node.name, node)
        if kind == "neg":
            q = self.eval(node.left)
            return Quantity(-q.value, q.dim)
        if kind == "call":
            args = [self.eval(a) for a in node.args]
            return self._call(node, args)
        if kind == "bin":
            return self._bin(node)
        if kind == "pow":
            return self._pow(node)
        raise EngineError(f"内部错误：未知节点 {kind}", node.start, node.end)

    # -- 二元运算 -----------------------------------------------------------
    def _bin(self, node: Node) -> Quantity:
        left = self.eval(node.left)
        right = self.eval(node.right)
        op = node.op
        if op in ("+", "-"):
            if left.dim != right.dim:
                word = "加" if op == "+" else "减"
                raise EngineError(
                    f"量纲不一致：「{dim_str(left.dim)}」与「{dim_str(right.dim)}」不能相{word}",
                    node.right.start, node.right.end, "dim_mismatch",
                )
            value = left.value + right.value if op == "+" else left.value - right.value
            return Quantity(value, left.dim)
        if op == "*":
            return Quantity(left.value * right.value, _dim_add(left.dim, right.dim))
        if op == "/":
            if right.value == 0:
                raise EngineError("除数为零", node.right.start, node.right.end, "value")
            return Quantity(left.value / right.value, _dim_sub(left.dim, right.dim))
        raise EngineError(f"内部错误：未知运算符 {op}", node.start, node.end)

    # -- 幂运算 -------------------------------------------------------------
    def _pow(self, node: Node) -> Quantity:
        base = self.eval(node.left)
        expo = self.eval(node.right)
        if expo.dim != DIMLESS:
            raise EngineError(
                f"幂指数必须无量纲（实际量纲为 {dim_str(expo.dim)}）",
                node.right.start, node.right.end, "bad_exponent",
            )
        k = Fraction(expo.value).limit_denominator(1_000_000)
        try:
            value = base.value ** float(k)
        except ZeroDivisionError:
            raise EngineError("零的负数次幂无意义", node.start, node.end, "value") from None
        except OverflowError:
            raise EngineError("数值溢出", node.start, node.end, "value") from None
        if isinstance(value, complex):
            raise EngineError("负数不能取分数次幂", node.start, node.end, "value")
        return Quantity(value, _dim_scale(base.dim, k))

    # -- 函数调用 -----------------------------------------------------------
    def _call(self, node: Node, args: list[Quantity]) -> Quantity:
        name = node.name
        meta = FUNCTIONS[name]
        if len(args) != meta["arity"]:
            raise EngineError(
                f"函数 {name} 需要 {meta['arity']} 个参数，实际给了 {len(args)} 个",
                node.start, node.end, "arity",
            )
        arg_node = node.args[0]
        q = args[0]
        if name == "sqrt":
            if q.value < 0:
                raise EngineError("sqrt 的参数不能为负数", arg_node.start, arg_node.end, "value")
            return Quantity(math.sqrt(q.value), _dim_scale(q.dim, Fraction(1, 2)))
        if name == "abs":
            return Quantity(abs(q.value), q.dim)
        if q.dim != DIMLESS:
            raise EngineError(
                f"函数 {name} 的参数必须无量纲（实际量纲为 {dim_str(q.dim)}）",
                arg_node.start, arg_node.end, "dimless_arg",
            )
        fn = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "exp": math.exp, "ln": math.log, "log": math.log10, "log2": math.log2,
        }[name]
        try:
            return Quantity(fn(q.value), DIMLESS)
        except ValueError:
            raise EngineError(f"函数 {name} 的参数超出定义域", arg_node.start, arg_node.end, "value") from None
        except OverflowError:
            raise EngineError("数值溢出", arg_node.start, arg_node.end, "value") from None


# ---------------------------------------------------------------------------
# 顶层分析入口
# ---------------------------------------------------------------------------

def parse_src(src: str) -> Node:
    """词法 + 语法分析（不做求值），便于复用 AST 反复重算。"""
    return Parser(tokenize(src)).parse()


def _parse_and_eval(src: str, resolve) -> Quantity:
    return Evaluator(resolve).eval(parse_src(src))


def _node_names(node: Node) -> set[str]:
    """收集 AST 中出现的全部标识符。"""
    names: set[str] = set()

    def walk(n: Node) -> None:
        if n.kind == "name":
            names.add(n.name)
        if n.left is not None:
            walk(n.left)
        if n.right is not None:
            walk(n.right)
        for a in n.args:
            walk(a)

    walk(node)
    return names


def _unit_only_resolver(variable_names: set[str], *, var_ref_message: str | None = None):
    """仅解析内置常量与单位（不确定度等纯数值输入中不允许引用变量）。"""
    message = var_ref_message or (
        "不确定度中不能引用变量「{name}」，标准不确定度必须是带单位的独立测量值"
    )

    def resolve(name: str, node: Node) -> Quantity:
        if name in CONSTANTS:
            value, d, _ = CONSTANTS[name]
            return Quantity(value, d)
        if name in UNITS:
            d, factor, _, _ = UNITS[name]
            return Quantity(factor, d)
        if name in variable_names:
            raise EngineError(message.format(name=name), node.start, node.end, "uncert_ref")
        raise EngineError(f"未定义的单位或常量「{name}」", node.start, node.end, "undefined")
    return resolve


def _make_var_resolver(variables: dict[str, str], var_ast: dict[str, Node],
                       var_qty: dict[str, Quantity], var_err: dict[str, EngineError],
                       stack: list[str], overrides: dict[str, Quantity] | None = None,
                       propagate_inner: bool = False):
    """构造变量解析闭包；overrides 中的变量直接取给定 Quantity（用于扰动）。

    propagate_inner=True 时（扰动重算），传播变量错误会保留内部错误的
    kind / 片段位置 / source，便于精确定位到变量链中真正出错的表达式。
    """
    overrides = overrides or {}

    def wrap_inner(name: str, node: Node) -> EngineError:
        inner = var_err[name].error
        if propagate_inner:
            exc = EngineError(
                f"变量「{name}」计算出错：{inner.message}",
                inner.start, inner.end, inner.kind,
            )
            exc.error.source = inner.source
            return exc
        return EngineError(
            f"变量「{name}」计算出错：{inner.message}",
            node.start, node.end, "var_error",
        )

    def evaluate_variable(name: str) -> None:
        if name in var_qty or name in var_err:
            return
        stack.append(name)
        try:
            var_qty[name] = Evaluator(resolve).eval(var_ast[name])
        except EngineError as exc:
            exc.error.source = f"var:{name}"
            var_err[name] = exc
        except OverflowError:
            err = EngineError("数值溢出", 0, len(variables[name]), "value")
            err.error.source = f"var:{name}"
            var_err[name] = err
        finally:
            stack.pop()

    def resolve(name: str, node: Node) -> Quantity:
        if name in overrides:
            return overrides[name]
        if name in var_qty:
            return var_qty[name]
        if name in var_err:
            raise wrap_inner(name, node)
        if name in variables:
            if name in stack:
                chain = " → ".join(stack + [name])
                raise EngineError(f"变量循环引用：{chain}", node.start, node.end, "var_cycle")
            evaluate_variable(name)
            if name in var_qty:
                return var_qty[name]
            raise wrap_inner(name, node)
        if name in CONSTANTS:
            value, d, _ = CONSTANTS[name]
            return Quantity(value, d)
        if name in UNITS:
            d, factor, _, _ = UNITS[name]
            return Quantity(factor, d)
        raise EngineError(f"未定义的变量或单位「{name}」", node.start, node.end, "undefined")

    return resolve, evaluate_variable


def _perturb_error(name: str, direction: str, exc: EngineError,
                   errors: list[AnalysisError]) -> None:
    """记录扰动重算失败：保留原错误（量纲/定义域等）与定位，标注扰动来源后立即中止。"""
    errors.append(AnalysisError(
        f"对输入「{name}」做 {direction} 侧扰动重算时失败：{exc.error.message}。"
        "不确定度分析中止，不返回合成不确定度。",
        exc.error.start, exc.error.end, exc.error.kind,
        source=exc.error.source,
        extra={"perturbed": name, "direction": direction},
    ))


def _uncertainty_analysis(expression: str, variables: dict[str, str],
                          var_ast: dict[str, Node], var_qty: dict[str, Quantity],
                          uncertainties: dict[str, str], main_ast: Node,
                          nominal: Quantity, errors: list[AnalysisError],
                          ) -> dict | None:
    """以各独立直接输入为扰动源，双侧中心差分求一阶灵敏度，RSS 合成标准不确定度。

    任一校验或扰动重算失败，都会向 errors 追加错误并返回 None，
    绝不返回貌似有效的不确定度数值。
    """
    # ---- 1. 校验每个不确定度输入 -------------------------------------------
    sources: list[dict] = []
    u_resolve = _unit_only_resolver(set(variables))
    for name in sorted(uncertainties):
        src = uncertainties[name]
        if name not in variables:
            errors.append(AnalysisError(
                f"不确定度对应的变量「{name}」不存在（请先在变量列表中定义它）",
                0, max(len(name), 1), "uncert_unknown", f"uncert:{name}",
            ))
            continue
        try:
            uq = _parse_and_eval(src, u_resolve)
        except EngineError as exc:
            exc.error.source = f"uncert:{name}"
            errors.append(exc.error)
            continue
        except OverflowError:
            errors.append(AnalysisError(
                f"变量「{name}」的标准不确定度数值溢出",
                0, len(src), "value", f"uncert:{name}",
            ))
            continue
        if not math.isfinite(uq.value):
            errors.append(AnalysisError(
                f"变量「{name}」的标准不确定度不是有限数值",
                0, len(src), "value", f"uncert:{name}",
            ))
            continue
        if uq.value < 0:
            errors.append(AnalysisError(
                f"变量「{name}」的标准不确定度不能为负数（实际为 {uq.value:g}）",
                0, len(src), "uncert_negative", f"uncert:{name}",
            ))
            continue
        if name in var_qty and uq.dim != var_qty[name].dim:
            errors.append(AnalysisError(
                f"变量「{name}」的标准不确定度量纲为「{dim_str(uq.dim)}」，"
                f"与该变量量纲「{dim_str(var_qty[name].dim)}」不一致",
                0, len(src), "uncert_dim", f"uncert:{name}",
            ))
            continue
        # 只有“直接输入”（定义式不引用任何用户变量）才能作为独立扰动源
        ast = var_ast.get(name)
        referenced = (_node_names(ast) if ast is not None else set()) & set(variables)
        if referenced:
            errors.append(AnalysisError(
                f"变量「{name}」引用了其他变量（{ '、'.join(sorted(referenced)) }），"
                "不是直接测量输入，不能单独填写标准不确定度；请为其引用的直接输入填写",
                0, max(len(variables[name]), 1), "uncert_indirect", f"uncert:{name}",
            ))
            continue
        sources.append({"name": name, "u_si": uq.value, "dim": uq.dim})

    if errors:
        return None

    # ---- 2. 零标称值也要有有效步长：相对步长 + 由不确定度/量级托底 ------------
    STEP_REL = 1e-6

    def step_for(src: dict) -> float:
        x0 = var_qty[src["name"]].value
        scale = max(abs(x0), src["u_si"])
        h = STEP_REL * scale
        if not math.isfinite(h) or h <= 0.0:
            h = STEP_REL
        return h

    # ---- 3. 逐源双侧扰动：完整变量链 + 主表达式全部重算 ----------------------
    contributions: list[dict] = []
    variance = 0.0
    for src in sources:
        name = src["name"]
        u_si = src["u_si"]
        x0 = var_qty[name].value
        h = step_for(src)

        def rerun(direction: str, xv: float) -> Quantity:
            q_overrides = {name: Quantity(xv, var_qty[name].dim)}
            qty: dict[str, Quantity] = {}
            err_map: dict[str, EngineError] = {}
            stk: list[str] = []
            resolver, _ = _make_var_resolver(
                variables, var_ast, qty, err_map, stk, q_overrides,
                propagate_inner=True)
            try:
                out = Evaluator(resolver).eval(main_ast)
            except OverflowError:
                raise EngineError("扰动重算数值溢出",
                                  0, len(expression), "uncert_eval") from None
            if not math.isfinite(out.value):
                raise EngineError("扰动重算结果不是有限数值",
                                  0, len(expression), "uncert_eval")
            if out.dim != nominal.dim:
                raise EngineError(
                    f"扰动后结果量纲变为「{dim_str(out.dim)}」，"
                    f"与标称量纲「{dim_str(nominal.dim)}」不一致",
                    0, len(expression), "uncert_eval")
            return out

        try:
            qp = rerun("+", x0 + h)
        except EngineError as exc:
            _perturb_error(name, "+", exc, errors)
            return None
        try:
            qm = rerun("-", x0 - h)
        except EngineError as exc:
            _perturb_error(name, "-", exc, errors)
            return None

        deriv = (qp.value - qm.value) / (2.0 * h)
        cu = abs(deriv) * u_si
        variance += cu * cu
        if u_si > 0.0:
            contributions.append({
                "name": name,
                "u_si": u_si,
                "step_si": h,
                "sensitivity": deriv,
                "contribution_si": cu,
            })

    # ---- 4. 一阶平方和（RSS）合成 -------------------------------------------
    uc = math.sqrt(variance)
    return {
        "method": "central_difference_rss",
        "step_relative": STEP_REL,
        "si_value": uc,
        "si_unit": si_unit_str(nominal.dim),
        "sources": contributions,
        "target": None,  # 由目标单位换算阶段填充
    }


def _normalize_inputs(expression: str, variables: dict | None,
                      uncertainties: dict | None) -> tuple[str, dict, dict]:
    """规范化 analyze / goal_seek 的表达式、变量与不确定度输入。"""
    expression = expression or ""
    variables = {
        str(k).strip(): (v if isinstance(v, str) else str(v))
        for k, v in (variables or {}).items() if str(k).strip()
    }
    uncertainties = {
        str(k).strip(): (v if isinstance(v, str) else str(v)).strip()
        for k, v in (uncertainties or {}).items()
        if str(k).strip() and (v if isinstance(v, str) else str(v)).strip()
    }
    return expression, variables, uncertainties


def _prepare_analysis(expression: str, variables: dict, uncertainties: dict
                      ) -> dict:
    """执行完整的常规校验：预解析变量/主表达式、求值整个变量链并求主表达式。

    返回准备阶段全部产物（AST、缓存、解析闭包、标称结果、错误与同名警告），
    供 analyze（不确定度分析）与 goal_seek（反求）共用，保证两种入口
    沿用完全一致的量纲与定义域规则。
    """
    errors: list[AnalysisError] = []
    # 变量名与内置单位/常量同名时给出非致命警告（变量优先级更高）
    warnings = []
    for name in variables:
        if name in UNITS:
            warnings.append({"message": f"变量「{name}」与内置单位同名，所有表达式中的 {name} 将按变量解析",
                             "name": name, "kind": "shadow_unit"})
        elif name in CONSTANTS:
            warnings.append({"message": f"变量「{name}」与内置常量同名，所有表达式中的 {name} 将按变量解析",
                             "name": name, "kind": "shadow_constant"})

    # 预解析全部变量与主表达式（AST 可供扰动 / 反求重算复用）
    var_ast: dict[str, Node] = {}
    for name, src in variables.items():
        try:
            var_ast[name] = parse_src(src)
        except EngineError as exc:
            exc.error.source = f"var:{name}"
            errors.append(exc.error)

    var_qty: dict[str, Quantity] = {}
    var_err: dict[str, EngineError] = {}
    stack: list[str] = []
    resolver, evaluate_variable = _make_var_resolver(
        variables, var_ast, var_qty, var_err, stack)

    # 先求值全部变量（即使未被引用，也能暴露变量自身的错误）
    for name in variables:
        if name not in var_ast:
            continue
        evaluate_variable(name)
        if name in var_err and var_err[name].error not in errors:
            errors.append(var_err[name].error)

    # 主表达式
    result: Quantity | None = None
    main_ast: Node | None = None
    try:
        main_ast = parse_src(expression)
    except EngineError as exc:
        exc.error.source = "expression"
        errors.append(exc.error)
    except OverflowError:
        err = EngineError("数值溢出", 0, len(expression), "value")
        err.error.source = "expression"
        errors.append(err.error)

    if main_ast is not None:
        try:
            result = Evaluator(resolver).eval(main_ast)
            if not math.isfinite(result.value):
                raise EngineError("计算结果不是有限数值", 0, len(expression), "value")
        except EngineError as exc:
            exc.error.source = "expression"
            errors.append(exc.error)
            result = None
        except OverflowError:
            err = EngineError("数值溢出", 0, len(expression), "value")
            err.error.source = "expression"
            errors.append(err.error)
            result = None

    return {
        "variables": variables,
        "uncertainties": uncertainties,
        "warnings": warnings,
        "errors": errors,
        "var_ast": var_ast,
        "var_qty": var_qty,
        "resolver": resolver,
        "result": result,
        "main_ast": main_ast,
    }


def analyze(expression: str, variables: dict | None = None,
            target_unit: str | None = None,
            uncertainties: dict | None = None) -> dict:
    """分析表达式，返回可 JSON 序列化的结果字典。

    即使存在错误也会尽量返回已算出的部分；errors 中每项都带有
    source（expression / target / var:<名称> / uncert:<名称>）与 (start, end) 片段位置。
    uncertainties 为 {变量名: 带单位的标准不确定度}，仅可标注在直接输入变量上。
    """
    expression, variables, uncertainties = _normalize_inputs(
        expression, variables, uncertainties)
    prep = _prepare_analysis(expression, variables, uncertainties)
    errors: list[AnalysisError] = prep["errors"]
    warnings = prep["warnings"]
    var_ast = prep["var_ast"]
    var_qty = prep["var_qty"]
    resolver = prep["resolver"]
    result = prep["result"]
    main_ast = prep["main_ast"]

    # 不确定度分析：仅在标称计算完全成功时进行；失败则追加错误且不产出数值
    uncertainty_block = None
    if result is not None and not any(e.source == "expression" for e in errors) \
            and not any(e.source.startswith("var:") for e in errors) and uncertainties:
        uncertainty_block = _uncertainty_analysis(
            expression, variables, var_ast, var_qty, uncertainties,
            main_ast, result, errors)

    # 目标单位换算（标称值与不确定度一起换算）
    target = None
    if target_unit and target_unit.strip():
        try:
            tq = _parse_and_eval(target_unit.strip(), resolver)
            if tq.value == 0:
                raise EngineError("目标单位不能为零", 0, len(target_unit), "value")
            if result is not None:
                if tq.dim != result.dim:
                    raise EngineError(
                        f"目标单位量纲为「{dim_str(tq.dim)}」，与结果量纲「{dim_str(result.dim)}」不一致",
                        0, len(target_unit), "target_dim",
                    )
                target = {"unit": target_unit.strip(), "value": result.value / tq.value}
                if uncertainty_block is not None:
                    uncertainty_block["target"] = {
                        "unit": target_unit.strip(),
                        "value": uncertainty_block["si_value"] / tq.value,
                    }
        except EngineError as exc:
            exc.error.source = "target"
            errors.append(exc.error)

    response: dict = {
        "ok": not errors,
        "errors": [e.to_dict() for e in errors],
        "warnings": warnings,
        "result": None,
    }
    if result is not None:
        response["result"] = {
            "si_value": result.value,
            "dimension": [format_fraction(x) for x in result.dim],
            "dimension_names": DIM_NAMES,
            "dimension_str": dim_str(result.dim),
            "si_unit": si_unit_str(result.dim),
            "derived": derived_matches(result.dim),
            "target": target,
            "uncertainty": uncertainty_block,
        }
    return response


# ---------------------------------------------------------------------------
# 反求（goal seek）：二分法求使主表达式达到目标值的直接输入变量
# ---------------------------------------------------------------------------

# 反求固定参数：端点容差系数与最大二分次数
_GOAL_TOL_SCALE = 1e-8
_GOAL_MAX_ITER = 80


def _pure_unit_resolver():
    """仅解析内置常量与单位（反求的变量单位 / 目标均为纯带单位数值）。"""
    def resolve(name: str, node: Node) -> Quantity:
        if name in CONSTANTS:
            value, d, _ = CONSTANTS[name]
            return Quantity(value, d)
        if name in UNITS:
            d, factor, _, _ = UNITS[name]
            return Quantity(factor, d)
        raise EngineError(f"未定义的单位或常量「{name}」", node.start, node.end, "undefined")
    return resolve


def goal_seek(expression: str, variables: dict | None, variable: str | None,
              unit: str | None, lower: float, upper: float,
              target: str | None, max_iter: int = _GOAL_MAX_ITER) -> dict:
    """反求直接输入变量 ``variable``，使主表达式达到带单位的目标结果 ``target``。

    流程：
      1. 先跑与 /api/check 完全一致的常规校验，任一既有规则失败则原样返回错误；
      2. 校验反求参数（变量存在且为直接输入、单位同量纲、有限上下界且下界<上界、
         目标与主表达式同量纲）；
      3. 每次试算以候选 SI 值覆盖该变量并重算整个下游变量链与主表达式；
      4. 端点已达标直接返回；否则仅在两端残差异号时二分，直到
         |残差| ≤ 1e-8·max(1, |目标 SI 值|) 或区间无法继续收缩；
      5. 最多 max_iter 次（默认 80）；区间不包根 / 中途求值失败 / 超限未收敛
         均返回明确错误且 result 为 null。
    """
    expression, variables, _ = _normalize_inputs(expression, variables, None)
    prep = _prepare_analysis(expression, variables, {})
    errors: list[AnalysisError] = prep["errors"]
    warnings = prep["warnings"]
    var_ast = prep["var_ast"]
    var_qty = prep["var_qty"]
    nominal = prep["result"]
    main_ast = prep["main_ast"]

    def fail_response(extra_errors: list[AnalysisError]) -> dict:
        return {
            "ok": False,
            "errors": [e.to_dict() for e in extra_errors],
            "warnings": warnings,
            "result": None,
        }

    # ---- 1. 常规校验（与 /api/check 相同的调用路径）必须先通过 ----------------
    if errors or nominal is None or main_ast is None:
        return fail_response(errors or [AnalysisError("标称表达式无法求值，不能反求",
                                                      0, len(expression), "solve")])

    var_name = (variable or "").strip()
    unit_src = (unit or "").strip()
    target_src = (target or "").strip()
    req_errors: list[AnalysisError] = []

    # ---- 2. 反求参数校验 ------------------------------------------------------
    var_dim: tuple | None = None
    if not var_name:
        req_errors.append(AnalysisError("必须指定待反求的变量名", 0, 1,
                                        "solve_var", "solve:variable"))
    elif var_name not in variables:
        req_errors.append(AnalysisError(
            f"待反求的变量「{var_name}」不存在（请先在变量列表中定义它）",
            0, max(len(var_name), 1), "solve_var_unknown", "solve:variable"))
    else:
        var_dim = var_qty[var_name].dim
        # 只能反求“直接输入”：定义式不引用任何其他用户变量
        referenced = _node_names(var_ast[var_name]) & set(variables)
        if referenced:
            req_errors.append(AnalysisError(
                f"变量「{var_name}」引用了其他变量（{'、'.join(sorted(referenced))}），"
                "是间接变量，不能作为反求对象；请反求其依赖的直接输入",
                0, max(len(variables[var_name]), 1),
                "solve_var_indirect", f"solve:variable"))

    if not unit_src:
        req_errors.append(AnalysisError("必须指定待求变量的取值单位", 0, 1,
                                        "solve_unit", "solve:unit"))
    unit_q: Quantity | None = None
    if unit_src:
        try:
            unit_q = _parse_and_eval(unit_src, _pure_unit_resolver())
            if not math.isfinite(unit_q.value) or unit_q.value <= 0.0:
                raise EngineError("变量单位必须是正的有限换算因子",
                                  0, len(unit_src), "value")
        except EngineError as exc:
            exc.error.source = "solve:unit"
            req_errors.append(exc.error)
        except OverflowError:
            req_errors.append(AnalysisError("变量单位数值溢出", 0, len(unit_src),
                                            "value", "solve:unit"))

    if unit_q is not None and var_dim is not None and unit_q.dim != var_dim:
        req_errors.append(AnalysisError(
            f"变量单位量纲为「{dim_str(unit_q.dim)}」，"
            f"与变量「{var_name}」的量纲「{dim_str(var_dim)}」不一致",
            0, len(unit_src), "solve_unit_dim", "solve:unit"))

    if not target_src:
        req_errors.append(AnalysisError("必须指定带单位的目标结果", 0, 1,
                                        "solve_target", "solve:target"))
    target_q: Quantity | None = None
    if target_src:
        try:
            target_q = _parse_and_eval(target_src, _pure_unit_resolver())
            if not math.isfinite(target_q.value):
                raise EngineError("目标结果不是有限数值", 0, len(target_src), "value")
        except EngineError as exc:
            exc.error.source = "solve:target"
            req_errors.append(exc.error)
        except OverflowError:
            req_errors.append(AnalysisError("目标结果数值溢出", 0, len(target_src),
                                            "value", "solve:target"))

    if target_q is not None and target_q.dim != nominal.dim:
        req_errors.append(AnalysisError(
            f"目标量纲为「{dim_str(target_q.dim)}」，"
            f"与主表达式量纲「{dim_str(nominal.dim)}」不一致",
            0, len(target_src), "solve_target_dim", "solve:target"))

    # 上下界：必须是有限数且下界 < 上界（按所给单位给出，换算为 SI）
    lo = hi = None
    if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
        req_errors.append(AnalysisError("下界必须是有限数", 0, 1,
                                        "solve_bound", "solve:bounds"))
    if not isinstance(upper, (int, float)) or not math.isfinite(float(upper)):
        req_errors.append(AnalysisError("上界必须是有限数", 0, 1,
                                        "solve_bound", "solve:bounds"))
    if (isinstance(lower, (int, float)) and isinstance(upper, (int, float))
            and math.isfinite(float(lower)) and math.isfinite(float(upper))):
        if float(lower) >= float(upper):
            req_errors.append(AnalysisError(
                f"下界（{lower:g}）必须严格小于上界（{upper:g}）",
                0, 1, "solve_bound_order", "solve:bounds"))
        elif unit_q is not None:
            lo = float(lower) * unit_q.value
            hi = float(upper) * unit_q.value
            if not (math.isfinite(lo) and math.isfinite(hi)):
                req_errors.append(AnalysisError("上下界换算到 SI 后溢出",
                                                0, 1, "solve_bound", "solve:bounds"))

    if req_errors:
        return fail_response(req_errors)

    # 以下不变量均已由上面的校验保证
    assert var_dim is not None and unit_q is not None and target_q is not None
    assert lo is not None and hi is not None
    tol = _GOAL_TOL_SCALE * max(1.0, abs(target_q.value))

    # ---- 3. 试算：候选值覆盖变量，重算完整变量链 + 主表达式 -------------------
    def trial(x_si: float, where: str) -> tuple[Quantity | None, AnalysisError | None]:
        overrides = {var_name: Quantity(x_si, var_dim)}
        qty: dict[str, Quantity] = {}
        err_map: dict[str, EngineError] = {}
        stk: list[str] = []
        resolver, _ = _make_var_resolver(
            variables, var_ast, qty, err_map, stk, overrides,
            propagate_inner=True)
        try:
            out = Evaluator(resolver).eval(main_ast)
        except EngineError as exc:
            return None, AnalysisError(
                f"在区间{where}（{var_name} = {x_si / unit_q.value:g} {unit_src}）"
                f"重算失败：{exc.error.message}。反求中止，不返回解。",
                exc.error.start, exc.error.end, exc.error.kind,
                source=exc.error.source,
                extra={"solve_var": var_name, "trial_si": x_si, "where": where})
        except OverflowError:
            return None, AnalysisError(
                f"在区间{where}（{var_name} = {x_si / unit_q.value:g} {unit_src}）"
                "重算时数值溢出。反求中止，不返回解。",
                0, len(expression), "value", source="expression",
                extra={"solve_var": var_name, "trial_si": x_si, "where": where})
        if not math.isfinite(out.value):
            return None, AnalysisError(
                f"在区间{where}（{var_name} = {x_si / unit_q.value:g} {unit_src}）"
                "重算结果不是有限数值。反求中止，不返回解。",
                0, len(expression), "solve_eval", source="expression",
                extra={"solve_var": var_name, "trial_si": x_si, "where": where})
        if out.dim != nominal.dim:
            return None, AnalysisError(
                f"在区间{where}重算后量纲变为「{dim_str(out.dim)}」，"
                f"与标称量纲「{dim_str(nominal.dim)}」不一致。反求中止，不返回解。",
                0, len(expression), "solve_eval", source="expression",
                extra={"solve_var": var_name, "trial_si": x_si, "where": where})
        return out, None

    # ---- 4. 端点求值：达标即返回，否则要求残差异号 ----------------------------
    def build_result(x_si: float, out: Quantity, iterations: int) -> dict:
        residual = out.value - target_q.value
        return {
            "ok": True,
            "errors": [],
            "warnings": warnings,
            "result": {
                "variable": var_name,
                "variable_si_value": x_si,
                "variable_si_unit": si_unit_str(var_dim),
                "value": x_si / unit_q.value,
                "unit": unit_src,
                "achieved_si_value": out.value,
                "achieved_si_unit": si_unit_str(out.dim),
                "achieved_dimension_str": dim_str(out.dim),
                "target_si_value": target_q.value,
                "residual_si": residual,
                "tolerance_si": tol,
                "iterations": iterations,
                "method": "bisection",
            },
        }

    out_lo, err = trial(lo, "下界")
    if err is not None:
        return fail_response([err])
    f_lo = out_lo.value - target_q.value
    if abs(f_lo) <= tol:
        return build_result(lo, out_lo, 0)

    out_hi, err = trial(hi, "上界")
    if err is not None:
        return fail_response([err])
    f_hi = out_hi.value - target_q.value
    if abs(f_hi) <= tol:
        return build_result(hi, out_hi, 0)

    # 端点均未达标：残差必须异号（f==0 时上面的容差检查已返回端点）
    if f_lo * f_hi > 0.0:
        return fail_response([AnalysisError(
            f"区间不包根：下界处残差为 {f_lo:g}，上界处残差为 {f_hi:g}（SI），"
            "两端必须异号才能二分；请调整上下界或目标值",
            0, 1, "solve_no_bracket", "solve:bounds",
            extra={"residual_lower_si": f_lo, "residual_upper_si": f_hi})])

    # ---- 5. 二分求根 ----------------------------------------------------------
    iterations = 0
    while iterations < max_iter:
        mid = 0.5 * (lo + hi)
        # 浮点区间已无法继续收缩：残差达标才算成功，否则按未收敛处理
        if mid == lo or mid == hi:
            return fail_response([AnalysisError(
                "反求区间已无法继续收缩（浮点数达到精度极限）且残差仍超过容差",
                0, 1, "solve_no_convergence", "solve:bounds",
                extra={"iterations": iterations,
                       "residual_si": f_lo if abs(f_lo) <= abs(f_hi) else f_hi})])
        out_mid, err = trial(mid, "内")
        if err is not None:
            err.extra = dict(err.extra or {})
            err.extra["iterations"] = iterations
            return fail_response([err])
        f_mid = out_mid.value - target_q.value
        iterations += 1
        if abs(f_mid) <= tol:
            return build_result(mid, out_mid, iterations)
        if f_lo * f_mid < 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo, out_lo = mid, f_mid, out_mid

    return fail_response([AnalysisError(
        f"二分 {max_iter} 次后仍未收敛（当前残差 {f_lo:g}，容差 {tol:g}）",
        0, 1, "solve_no_convergence", "solve:bounds",
        extra={"iterations": iterations, "residual_si": f_lo, "tolerance_si": tol})])
