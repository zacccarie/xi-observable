"""Re-run des 5 trajectoires grokking en SAUVANT les checkpoints.

But : ne plus jamais re-entraîner. Tous les observables futurs (Γ, TC, probes
poids, restricted-loss...) se calculent offline sur ce qui est sauvé ici.

Sauvé tous les 50 steps, par run (grok_ckpts/run_wd{wd}_s{seed}/) :
  step_XXXXX.npz   : acts_test (512,128) f16, acts_train (512,128) f16
  weights_XXXXX.pt : state_dict fp16
  traj.json        : métriques par checkpoint (train_acc, val_acc, weight_norm)

~330 s/run CPU × 5 runs. Disque ~400 MB.
"""
import sys
import os
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/zac/claude-code-projects/critical-grownet")

from grok import make_data, GrokGPT, DEVICE

CKPT_ROOT = os.path.join(HERE, "grok_ckpts")


def run_and_save(seed, weight_decay, steps=3000, every=50, p=97):
    outdir = os.path.join(CKPT_ROOT, f"run_wd{weight_decay}_s{seed}")
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xte, Yte, vocab = make_data(p, seed)
    model = GrokGPT(vocab, p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=weight_decay, betas=(0.9, 0.98))
    probe_tr, probe_te = Xtr[:512], Xte[:512]

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def acts(Xs):
        model.eval()
        _, la = model(Xs, want_last_act=True)
        model.train()
        return la.cpu().numpy().astype(np.float16)

    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        if step % every == 0:
            np.savez_compressed(os.path.join(outdir, f"step_{step:05d}.npz"),
                                acts_test=acts(probe_te), acts_train=acts(probe_tr))
            torch.save({k: v.half() for k, v in model.state_dict().items()},
                       os.path.join(outdir, f"weights_{step:05d}.pt"))
            wn = float(sum(p_.norm().item() ** 2
                           for p_ in model.parameters()) ** 0.5)
            traj.append({"step": step, "train_acc": acc(Xtr, Ytr),
                         "val_acc": acc(Xte, Yte), "weight_norm": wn})
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    meta = {"seed": seed, "wd": weight_decay, "t_grok": t_grok,
            "traj": traj, "elapsed_s": time.time() - t0}
    with open(os.path.join(outdir, "traj.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  wd={weight_decay} s{seed} : t_grok={t_grok} "
          f"({meta['elapsed_s']:.0f}s) → {outdir}")
    return meta


if __name__ == "__main__":
    t0 = time.time()
    print("[checkpoint runs]")
    for seed in range(3):
        run_and_save(seed, weight_decay=1.0)
    for seed in range(2):
        run_and_save(seed, weight_decay=0.0)
    print(f"total {time.time() - t0:.0f}s")
