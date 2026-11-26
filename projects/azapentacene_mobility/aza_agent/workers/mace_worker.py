"""Persistent MACE-OFF worker (run in the BOMLIP_CSP env). JSON lines on stdin -> JSON lines on fd 3-style
private stdout; everything else printed by libraries goes to stderr.

Requests:
  {"cmd": "energies", "cifs": [path, ...]}            -> {"results": [{"path", "energy_eV", "natoms"} | {"path", "error"}]}
  {"cmd": "molecules", "molecules": [{"symbols", "positions"}, ...]} -> {"energies_eV": [...]}
  {"cmd": "gas", "symbols": [...], "positions": [[x,y,z],...], "fmax": 0.01}
                                                     -> {"energy_eV", "steps", "converged"}
  {"cmd": "relax", "cifs": [...], "out_dir": dir, "steps": 100, "fmax": 0.05, "cell": true, "dtype": "float32",
   "workers": 4}  -> {"results": [{"path", "out_path", "E0_eV", "E_eV", "steps", "converged", "fmax", "V0", "V",
                                   "seconds"} | {"path", "error"}]}
     quick relaxation (ASE LBFGS maxstep 0.08; FrechetCellFilter P=0 if cell) in a pool of spawned processes
     sharing the GPU (benchmark 2026-11-25: float32 = same minima as float64, 2.7x faster on the L4).
Energies for ranking always come from the float64 "energies" single points.
"""
import json
import os
import sys
import time
import traceback

_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr


_R = {}


def _relax_init(model, dtype, threads):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from mace.calculators import mace_off
    torch.set_num_threads(threads)
    _R["calc"] = mace_off(model=model, device="cuda", default_dtype=dtype)


def _relax_one(job):
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.io import write
    from ase.optimize import LBFGS
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    path, out_path, steps, fmax, cell = job
    t0 = time.time()
    try:
        at = AseAtomsAdaptor.get_atoms(Structure.from_file(path))
        at.calc = _R["calc"]
        v0 = float(at.get_volume())
        e0 = float(at.get_potential_energy())
        opt = LBFGS(FrechetCellFilter(at, scalar_pressure=0.0) if cell else at, maxstep=0.08, logfile=None)
        conv = bool(opt.run(fmax=fmax, steps=steps))
        f = float(np.linalg.norm(at.get_forces(), axis=1).max())
        e1 = float(at.get_potential_energy())
        write(out_path, at, format="cif")
        return dict(path=path, out_path=out_path, E0_eV=e0, E_eV=e1, steps=int(opt.nsteps), converged=conv,
                    fmax=f, V0=v0, V=float(at.get_volume()), seconds=round(time.time() - t0, 2))
    except Exception as e:  # record, never drop
        return dict(path=path, error=f"{type(e).__name__}: {e}"[:300], seconds=round(time.time() - t0, 2))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    from ase import Atoms
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    def read(path):  # pymatgen handles CIF symmetry blocks that ASE rejects
        return AseAtomsAdaptor.get_atoms(Structure.from_file(path))
    from ase.optimize import BFGS
    from mace.calculators import mace_off

    t0 = time.time()
    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    _out.write(json.dumps(dict(ready=True, model=f"MACE-OFF23_{args.model}", dtype=args.dtype,
                               load_seconds=round(time.time() - t0, 2))) + "\n")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req["cmd"] == "energies":
                res = []
                for p in req["cifs"]:
                    try:
                        at = read(p)
                        at.calc = calc
                        res.append(dict(path=p, energy_eV=float(at.get_potential_energy()), natoms=len(at)))
                    except Exception as e:  # record, never drop
                        res.append(dict(path=p, error=f"{type(e).__name__}: {e}"))
                resp = dict(results=res)
            elif req["cmd"] == "gas":
                at = Atoms(req["symbols"], positions=req["positions"])
                at.calc = calc
                opt = BFGS(at, logfile=None)
                conv = opt.run(fmax=req.get("fmax", 0.01), steps=req.get("steps", 500))
                resp = dict(energy_eV=float(at.get_potential_energy()), steps=opt.nsteps, converged=bool(conv))
            elif req["cmd"] == "molecules":  # frozen-geometry single points of isolated molecules
                es = []
                for m in req["molecules"]:
                    at = Atoms(m["symbols"], positions=m["positions"])
                    at.calc = calc
                    es.append(float(at.get_potential_energy()))
                resp = dict(energies_eV=es)
            elif req["cmd"] == "relax":
                import multiprocessing as mp
                key = (req.get("dtype", "float32"), int(req.get("workers", 4)))
                if _R.get("pool_key") != key:
                    if _R.get("pool"):
                        _R["pool"].close()
                    _R["pool"] = mp.get_context("spawn").Pool(key[1], initializer=_relax_init,
                                                              initargs=(args.model, key[0], 1))
                    _R["pool_key"] = key
                os.makedirs(req["out_dir"], exist_ok=True)
                jobs = [(p, os.path.join(req["out_dir"], os.path.basename(p)), int(req.get("steps", 100)),
                         float(req.get("fmax", 0.05)), bool(req.get("cell", True))) for p in req["cifs"]]
                resp = dict(results=_R["pool"].map(_relax_one, jobs, chunksize=1))
            elif req["cmd"] == "quit":
                if _R.get("pool"):
                    _R["pool"].close()
                break
            else:
                resp = dict(error=f"unknown cmd {req['cmd']}")
        except Exception as e:
            resp = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        _out.write(json.dumps(resp) + "\n")


if __name__ == "__main__":
    main()
