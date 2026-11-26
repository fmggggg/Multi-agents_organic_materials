"""CPU-only checks of the chemical space and the duplicate gate (pytest, or `python tests/test_space.py`)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem  # noqa: E402

from aza_agent import space  # noqa: E402
from aza_agent.config import KNOWN_CSV  # noqa: E402

EXPECTED = {"AAA", "AAB", "AAL", "ABA", "ABL", "ALA", "ALB", "ALL", "LAL", "LLL"}


def test_ten_skeletons_with_14_sites():
    sks = space.skeletons()
    assert set(sks) == EXPECTED
    for code in sks:
        mol = Chem.MolFromSmiles(sks[code].smiles)
        assert mol.GetNumAtoms() == 22
        assert sum(a.GetTotalNumHs() for a in mol.GetAtoms()) == 14


def test_build_identify_round_trip():
    rng = random.Random(0)
    for code in sorted(space.skeletons()):
        for _ in range(20):
            sites = sorted(rng.sample(range(1, 15), rng.randint(1, 8)))
            smi = space.build_from_sites(code, sites)
            got_code, got_sites = space.identify(Chem.MolFromSmiles(smi))
            assert got_code == code
            # identify returns the canonical (smallest) symmetry-equivalent site set: same molecule
            assert Chem.CanonSmiles(space.build_from_sites(got_code, got_sites)) == Chem.CanonSmiles(smi)


def test_gate_rules():
    no_n = space.check("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1")  # pentacene: no N
    assert not no_n["ok"]
    dup = space.check(space.build_from_sites("LLL", [6, 13]), campaign_graph_keys=(), campaign_aroma_keys=())
    if dup["ok"]:  # not in the reference set: must be rejected once tested in the campaign
        again = space.check(dup["smiles"], {dup["graph_key"]}, {dup["aroma_key"]})
        assert not again["ok"] and again["reason"] == "duplicate_already_tested"


def test_known_set_recognised():
    if not Path(KNOWN_CSV).exists():
        print("skip: reference CSV not present", KNOWN_CSV)
        return
    import pandas as pd
    smiles = pd.read_csv(KNOWN_CSV)["Bonded_SMILES"].dropna().tolist()[:300]
    in_space = [s for s in smiles if space.identify(Chem.MolFromSmiles(s))[0]]
    assert in_space
    assert all(space.check(s)["reason"] == "duplicate_known_dataset" for s in in_space)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
