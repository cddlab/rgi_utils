"""Config expression DSL: parse an ``energy: "<formula>"`` string into a safe AST and
evaluate it against a ``RestraintContext``.

A formula is a Python *expression* over the vocabulary (``context.CALL_NAMES``) and named
atom-group selections, e.g. ``"(distance(A, B) - distance(C, D))**2"`` or
``"flat_bottomed(angle(A,B,C), 1.4, 1.8)"``. It is parsed with ``ast.parse`` (NOT ``eval``)
and validated against a strict node whitelist, so no attribute access / subscripting /
lambdas / comprehensions / arbitrary calls are possible — only arithmetic, comparisons,
conditional expressions (``a if cond else b``), logical ``and``/``or``/``not``, and calls to
the vocabulary. The AST is built ONCE at setup (static), so evaluating it on the jax backend
traces to a fixed op graph (lax.scan-safe).

**Branching is elementwise and does NOT short-circuit.** ``a if cond else b`` is rewritten to
``where(cond, a, b)`` and ``and``/``or`` to ``&``/``|``, so *both* sides are always evaluated —
that is what keeps the closure traceable inside ``lax.scan`` (a real Python branch on a traced
value is impossible). Two consequences: the dead branch's NaN/inf poisons the *gradient* even
when the value is fine (the usual ``where`` trap), and the resolve pass sees the selections of
both branches (which is why a conditional formula resolves every group it mentions).

Identifiers: a bare ``Name`` (``A``) is a selection identifier resolved by name from the
entry's ``selections`` map; a string ``Constant`` (``"chain A"``) is a raw selection
string. Numbers are numeric ``Constant`` s.
"""

from __future__ import annotations

import ast
from functools import reduce

from rgi_utils.custom.context import CALL_NAMES

# arithmetic + boolean-combine (``&``/``|`` compose masks for where(); ``and``/``or`` below
# map onto the same two operators).
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
}
_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}
# ``and``/``or`` delegate to the operands' own ``&``/``|``, which numpy / torch / jax all
# overload elementwise on boolean arrays — so no backend dispatch is needed here.
_BOOLOPS = {
    ast.And: lambda a, b: a & b,
    ast.Or: lambda a, b: a | b,
}


def _logical_not(value):
    """Elementwise logical negation.

    ``~`` is the array negation, but on a Python ``bool`` it is *integer* inversion
    (``~True == -2``, which is truthy) — and comparisons DO fold to Python bools on the
    setup-time ``ResolveContext``, so that path is real, not hypothetical.
    """
    if isinstance(value, bool):
        return not value
    return ~value


_UNARYOPS = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a, ast.Not: _logical_not}

# AST node types permitted anywhere in a formula (everything else is rejected at parse).
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    *_BINOPS,
    *_BOOLOPS,
    *_CMPOPS,
    *_UNARYOPS,
)


def parse_formula(formula: str) -> ast.Expression:
    """Parse + validate a formula string into an AST. Raises ``ValueError`` on any
    disallowed construct (so a malformed / unsafe formula fails loudly at config time)."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"custom energy formula is not a valid expression: {exc}"
        ) from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"custom energy formula: disallowed construct "
                f"{type(node).__name__!r} (only arithmetic, comparisons, "
                f"'a if cond else b', and/or/not, and calls to "
                f"{sorted(CALL_NAMES)} over named selections are allowed)"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in CALL_NAMES:
                raise ValueError(
                    f"custom energy formula: only calls to {sorted(CALL_NAMES)} are "
                    "allowed (no attributes, no arbitrary functions)"
                )
            if node.keywords:
                raise ValueError(
                    "custom energy formula: keyword args are not supported"
                )
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float, str)
        ):
            raise ValueError(
                f"custom energy formula: constant {node.value!r} must be a number or a "
                "selection string"
            )
    return tree


def eval_formula(tree: ast.Expression, ctx):
    """Evaluate a parsed formula against ``ctx`` (a RestraintContext / ResolveContext)."""
    return _ev(tree.body, ctx)


def _ev(node, ctx):
    if isinstance(node, ast.Constant):
        return node.value  # number, or a selection string
    if isinstance(node, ast.Name):
        return node.id  # selection identifier (resolved by ctx._idx)
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_ev(node.operand, ctx))
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_ev(node.left, ctx), _ev(node.right, ctx))
    if isinstance(node, ast.BoolOp):
        # ``a and b and c`` is ONE BoolOp with three values, so fold rather than assume two.
        return reduce(
            _BOOLOPS[type(node.op)], (_ev(value, ctx) for value in node.values)
        )
    if isinstance(node, ast.IfExp):
        # Elementwise select — both branches are evaluated (see the module docstring).
        return ctx.where(
            _ev(node.test, ctx), _ev(node.body, ctx), _ev(node.orelse, ctx)
        )
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ValueError("custom energy formula: chained comparisons not supported")
        return _CMPOPS[type(node.ops[0])](
            _ev(node.left, ctx), _ev(node.comparators[0], ctx)
        )
    if isinstance(node, ast.Call):
        args = [_ev(a, ctx) for a in node.args]
        return getattr(ctx, node.func.id)(*args)
    raise ValueError(f"custom energy formula: cannot evaluate {type(node).__name__}")
