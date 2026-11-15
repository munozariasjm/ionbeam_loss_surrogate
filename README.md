# A differentiable surrogate for loss-dominated ion beam transport

Particle tracking through an electrostatic beamline is slow and not
differentiable. This replaces it with a likelihood that is both, by factorising
the simulator response into the mass that survives and the shape of what
survives:

```
p(ion at x, plane k | u) = T_k(u) · p(x | u, s_k, survived)
                            survival     conditional phase space
```

`u` are 22 electrode voltages; the ten detector planes sit at measured arc
lengths from 0 to 1121 mm. `T_k` is a discrete-time survival model read plane by
plane from a GRU; `p(x | ·)` is a conditional rational-quadratic neural spline
flow over the six-dimensional phase space, conditioned on arc length. The two
are trained separately and multiplied at inference.

Roughly half of all settings deliver no beam at all to the exit plane, which is
the regime this is built for.

## Install

```bash
pip install -e ".[data,dev]"
```

## Run

Gradient ascent on transmission:

```bash
python scripts/optimize.py
python scripts/optimize.py --width-target 2.0
```

Reconstructing the beam where there is no detector, from survivor counts at
three planes:

```bash
python scripts/infer.py
```

Both print predictions, not measurements. Re-fly anything you intend to use.

## Reproduce

With the dataset in place (see below):

```bash
export BSIMS_DATA=/path/to/data

python scripts/evaluate.py --ensemble models --flow models/flow_s0.eqx \
    --dispersion models/dispersion.eqx

for k in $(seq 1 12); do
  python scripts/train_hazard.py --seed $k --bootstrap $k \
      --out models/hazard_ens_m$k.eqx
done
python scripts/train_flow.py --out models/flow_s0.eqx

python scripts/infer.py --data $BSIMS_DATA     # adds the exit image
pytest
```

A transmission head trains in about two minutes on one GPU, the flow in about
an hour.

# Data

The dataset paired along this project is publicly available in Zenodo: [10.5281/zenodo.24250916](https://zenodo.org/records/24250916)


# Cite

Please cite as:

```
@misc{munozarias2026differentiablesurrogatelossdominatedion,
      title={A Differentiable Surrogate for Loss-Dominated Ion Beam Transport},
      author={Jose M. Munoz-Arias and Laura Ruiz-Arango and Derick Gonzalez-Acevedo and Fabian C. Pastrana-Cruz and Jackson Hacias and Ronald F. Garcia Ruiz},
      year={2026},
      eprint={2610.08877},
      archivePrefix={arXiv},
      primaryClass={physics.ins-det},
      url={https://arxiv.org/abs/2610.08877},
}
```
