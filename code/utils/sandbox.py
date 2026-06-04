"""Restricted globals for executing LLM-generated verification snippets."""

import datetime
import math
import re

import pandas as pd

try:
    import z3
except Exception:  # pragma: no cover - z3 is optional for non-Z3 snippets.
    z3 = None


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed = {
        "datetime": datetime,
        "math": math,
        "pandas": pd,
        "pd": pd,
        "re": re,
    }
    if z3 is not None:
        allowed["z3"] = z3
    if level == 0 and name in allowed:
        return allowed[name]
    raise ImportError(f"Import of {name!r} is not allowed in verifier snippets")

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "hasattr": hasattr,
    "int": int,
    "AttributeError": AttributeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    "zip": zip,
    "__import__": _safe_import,
}


def build_exec_globals(extra=None):
    """Build an exec globals dict without import/open/eval/exec builtins."""
    globals_dict = {"__builtins__": SAFE_BUILTINS}
    if extra:
        globals_dict.update(extra)
    return globals_dict
