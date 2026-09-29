Faithful JAX kernel of HyperReservoirs (arXiv:2609.34847): two leaky ESNs with a bilinear/Kronecker readout trained by ridge regression.

## Install

```
python3 -m venv .venv && .venv/bin/pip install -e '.[tests]'
```

## Run

```
.venv/bin/pytest -q
.venv/bin/python train.py
```

## What does not match the paper

No paper-scale Lorenz/Rössler tables, no 22-run means, conceptor aperture sweep omitted or tiny, spectral-radius/leak not paper-tuned.
