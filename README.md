# Derivitator

A small Flask app that computes the symbolic derivative of a single-variable
function of `x`, using [SymPy](https://www.sympy.org/) for the actual math.

## Features

- Real symbolic differentiation (polynomials, trig, inverse trig, `e^x`,
  `ln`/`log`, products, quotients, chain rule)
- Implicit multiplication support (`2x`, `2sin(x)`, `x(x+1)`)
- `^` accepted for exponents (`x^2`)
- Friendly error messages for invalid syntax, unsupported variables, or
  undefined expressions (e.g. division by zero)

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

> Note: this needs to run as an actual local (or hosted) server — it won't
> work if you just open `templates/index.html` as a static file, since the
> page calls a `/differentiate` API route that only exists while `app.py`
> is running.

## Project structure

```
derivitator/
├── app.py              # Flask backend + SymPy parsing/differentiation
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt
└── README.md
```

## Notes / assumptions

- Only `x` is supported as a variable.
- `log(x)` and `ln(x)` are both treated as the natural logarithm.

## Roadmap

- [ ] Graph of the function and its derivative
