"""Persistent MCRT(i) mobility worker. Run inside MCRT_container.sif (apptainer exec --nv).
Loads the archived Jay-mobility checkpoint once and scores CIF batches on request.
Settings are those of MCRT/server_finetune_mobility_Jay_test.py (ALIGNN 4x256, no image,
mean 2.674, std 2.813, batch 1), as in the handoff mcrt_predict.py adapter.

Request:  {"cmd": "predict", "cifs": [path, ...], "work_dir": dir}
Response: {"results": [{"cif_id", "path", "mobility"}], "missing": [...], "seconds"}
"""
import json
import os
import sys
import time
import traceback

_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

MCRT_ROOT = os.environ.get("AZA_MCRT_ROOT") or "/mnt/scratch/users/ffmmgg/MCRT"
CKPT = os.environ.get("AZA_MCRT_CKPT") or (
    f"{MCRT_ROOT}/logs/finetune/mobility_Jay/pretraining_MCRT_seed0_from_epoch=29-step=33120/"
    "version_1/checkpoints/epoch=46-step=62134.ckpt")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--accelerator", default="gpu")
    args = ap.parse_args()

    sys.path.insert(0, MCRT_ROOT)
    os.environ.setdefault("DGLBACKEND", "pytorch")
    import copy
    import math
    import shutil
    import warnings

    warnings.filterwarnings("ignore")
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import Callback

    from MCRT import server_finetune_mobility_Jay_test as legacy
    from MCRT.config import config as _config
    from MCRT.modules.datamodule import Datamodule
    from MCRT.modules.McrtModule import MCRT_Module
    from MCRT.modules.utils import get_valid_config

    t0 = time.time()
    cfg = copy.deepcopy(_config())
    for key in ("max_epochs", "loss_names", "batch_size", "per_gpu_batchsize", "num_workers", "if_conv",
                "if_alignn", "alignn_num_conv", "alignn_hidden_dim", "pos_emb", "seed", "if_image",
                "test_only", "test_to_csv", "learning_rate", "mean", "std"):
        cfg[key] = getattr(legacy, key)
    cfg.update(downstream=legacy.downstream, load_path=args.checkpoint, accelerator=args.accelerator,
               devices=1, root_dataset="", log_dir="/tmp", test_only=True)
    pl.seed_everything(cfg["seed"])
    cfg = get_valid_config(cfg)
    model = MCRT_Module(cfg)
    torch.set_float32_matmul_precision("medium")
    _out.write(json.dumps(dict(ready=True, checkpoint=args.checkpoint, load_seconds=round(time.time() - t0, 2))) + "\n")

    class Collect(Callback):
        def __init__(self):
            self.ids, self.preds = [], []

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.preds.extend(outputs["regression_logits"].cpu().numpy().reshape(-1).tolist())
            self.ids.extend(outputs["cif_id"])

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req["cmd"] == "quit":
                break
            t1 = time.time()
            work = req["work_dir"]
            if os.path.exists(work):
                shutil.rmtree(work)
            os.makedirs(f"{work}/cifs")
            id2path = {}
            for p in req["cifs"]:
                cid = os.path.splitext(os.path.basename(p))[0]
                id2path[cid] = p
                os.symlink(os.path.abspath(p), f"{work}/cifs/{cid}.cif")
            ids = sorted(id2path)
            with open(f"{work}/dataset_split.json", "w") as h:
                json.dump(dict(train=[], val=[], test=ids), h)
            with open(f"{work}/mobility_labels.csv", "w") as h:
                h.writelines(f"{i},0.0\n" for i in ids)
            open(f"{work}/skip_cifs.txt", "w").close()
            with open(f"{work}/shard_ids.txt", "w") as h:
                h.write("\n".join(ids) + "\n")
            c = dict(cfg, root_dataset=work)
            dm = Datamodule(c)
            col = Collect()
            trainer = pl.Trainer(accelerator=args.accelerator, devices=1, precision=c["precision"], logger=False,
                                 enable_checkpointing=False, enable_progress_bar=False, callbacks=[col],
                                 deterministic=True)
            trainer.test(model, datamodule=dm, verbose=False)
            got = {}
            for cid, v in zip(col.ids, col.preds):
                cid = cid[:-4] if cid.endswith(".cif") else cid
                got[cid] = v
            res = [dict(cif_id=i, path=id2path[i], mobility=float(got[i])) for i in ids
                   if i in got and math.isfinite(got[i])]
            resp = dict(results=res, missing=[i for i in ids if i not in got or not math.isfinite(got[i])],
                        seconds=round(time.time() - t1, 2))
            shutil.rmtree(work, ignore_errors=True)
        except Exception as e:
            resp = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        _out.write(json.dumps(resp) + "\n")


if __name__ == "__main__":
    main()
