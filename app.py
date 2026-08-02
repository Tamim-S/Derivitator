"""
Derivitator backend — symbolic differentiation of single-variable functions of x.

Run:
    pip install flask sympy
    python app.py
Then open http://localhost:5000
"""
import re
import signal

import sympy
from flask import Flask, jsonify, render_template, request
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

app = Flask(__name__)

# Symbolic setup
X = sympy.Symbol("x", real=True)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,  # "2x", "2sin(x)", "x(x+1)" -> explicit *
    convert_xor,  # "^" -> "**"
)


def _natural_log(arg):
    # Both log(x) and ln(x) are treated as the natural logarithm.
    return sympy.log(arg)

# Terms dictionary
LOCAL_DICT = {
    "x": X,
    "e": sympy.E,
    "E": sympy.E,
    "pi": sympy.pi,
    "PI": sympy.pi,
    "ln": _natural_log,
    "log": _natural_log,
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
    "arcsin": sympy.asin,
    "arccos": sympy.acos,
    "arctan": sympy.atan,
}

# Characters allowed in a raw input string, checked before it ever reaches the
# parser. Defense-in-depth: even though sympy's parser uses a restricted
# namespace, we don't hand it anything outside basic math syntax.
_ALLOWED_INPUT = re.compile(r"^[0-9a-zA-Z\s\+\-\*\/\^\(\)\.\,\_]*$")
MAX_INPUT_LENGTH = 200

# Errors
class DifferentiationError(Exception):
    """Any user-facing failure: bad syntax, unsupported variable, etc."""

# Class for fractions
class FractionPrinter(StrPrinter):
    """
    sympy's default printer renders x**-1 as '1/x' but x**-2 as 'x**(-2)'.
    This subclass makes ALL negative integer powers print as a fraction,
    which reads far more naturally for a derivative — e.g. tan(x)' shows as
    "1/cos(x)^2" instead of "cos(x)**(-2)".
    """

    def _print_Pow(self, expr, rational=False):
        if expr.exp.is_negative and expr.exp.is_Integer and expr.exp != -1:
            positive_power = sympy.Pow(expr.base, -expr.exp)
            denominator = self.parenthesize(positive_power, PRECEDENCE["Mul"] - 1)
            return "1/%s" % denominator
        return super()._print_Pow(expr, rational=rational)


def _sympy_str(expr):
    return FractionPrinter().doprint(expr)


def _round_floats(text, precision=6):
    """Collapse sympy's long float literals (e.g. 5.00000000000000) down to
    a clean decimal (5), while still preserving real precision (2.5)."""

    def _clean(match):
        value = float(match.group(0))
        formatted = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return formatted or "0"

    return re.sub(r"\d+\.\d+", _clean, text)


def format_expression(expr):
    """Turn a sympy expression into calculator-style display text:
    x**2 -> x^2, 2*x -> 2x, 2*x*cos(x**2) -> 2x cos(x^2), etc.
    """
    text = _sympy_str(expr)
    text = text.replace("**", "^")

    # '*' is dropped for tight coefficient binding ("2x") and turned into a
    # space between separate factors ("2x cos(x^2)") for readability.
    rebuilt = []
    for i, ch in enumerate(text):
        if ch == "*":
            rebuilt.append("" if (i > 0 and text[i - 1].isdigit()) else " ")
        else:
            rebuilt.append(ch)
    text = "".join(rebuilt)

    text = text.replace("exp(", "e^(")
    text = re.sub(r"e\^\(([a-zA-Z0-9])\)", r"e^\1", text)  # e^(x) -> e^x
    text = text.replace("log(", "ln(")
    text = text.replace("Abs(", "abs(")
    text = text.replace("asin(", "arcsin(")
    text = text.replace("acos(", "arccos(")
    text = text.replace("atan(", "arctan(")
    text = _round_floats(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Creates an error for functions not of x, too long input, or syntax errors
def _validate_raw_input(raw):
    if not isinstance(raw, str):
        raise DifferentiationError("Please enter a function of x.")
    cleaned = raw.strip()
    if not cleaned:
        raise DifferentiationError("Please enter a function of x.")
    if len(cleaned) > MAX_INPUT_LENGTH:
        raise DifferentiationError("That function is too long.")
    if not _ALLOWED_INPUT.match(cleaned):
        raise DifferentiationError(
            "Only numbers, x, +, -, *, /, ^, parentheses, and function names "
            "(sin, cos, ln, sqrt, ...) are supported."
        )
    return cleaned

# Breaks down words into code for output purposes
def parse_function(raw):
    cleaned = _validate_raw_input(raw)
    try:
        expr = parse_expr(cleaned, local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)
    except Exception:
        raise DifferentiationError(
            "Could not parse that function. Check for balanced parentheses and "
            'that function names are followed by "(...)", e.g. cos(x).'
        )

    if expr.has(sympy.zoo, sympy.nan):
        raise DifferentiationError("That expression is undefined (division by zero).")

    extra_symbols = expr.free_symbols - {X}
    if extra_symbols:
        names = ", ".join(sorted(str(s) for s in extra_symbols))
        raise DifferentiationError(
            f"Only 'x' can be used as a variable (found: {names}). If you meant "
            "a function, make sure it has parentheses, e.g. cos(x)."
        )

    return expr

# Timeout error
class _TimeoutError(Exception):
    pass


def _simplify_with_timeout(expr, seconds=5):
    """Best-effort simplification with a timeout so one pathological input
    can't hang the server. Falls back to the unsimplified derivative."""
    if not hasattr(signal, "SIGALRM"):
        try:
            return sympy.simplify(expr)
        except Exception:
            return expr

    def _handler(signum, frame):
        raise _TimeoutError()

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return sympy.simplify(expr)
    except Exception:
        return expr
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

# Uses SymPy and finds the derivative
def differentiate(raw):
    """Returns (original_display, derivative_display). Raises DifferentiationError."""
    expr = parse_function(raw)

    try:
        derivative = sympy.diff(expr, X)
    except Exception:
        raise DifferentiationError("Could not differentiate that function.")

    derivative = _simplify_with_timeout(derivative)

    return format_expression(expr), format_expression(derivative)


# Routes

# Links app.py to index.html
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/differentiate", methods=["POST"])
def differentiate_route():
    data = request.get_json(silent=True) or {}
    raw = data.get("function", "")

    try:
        original, derivative = differentiate(raw)
    except DifferentiationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify(
            {"success": False, "error": "Something went wrong differentiating that function."}
        ), 500

    return jsonify({"success": True, "original": original, "derivative": derivative})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
