"""FP8 numerical stability diagnostics for Boltz-1 training.

Orchestrates 4 experiments:
  1. FP32 baseline
  2. BF16-mixed baseline (+ divergence check vs FP32)
  3. Naive FP8 – all nn.Linear replaced with te.Linear + fp8_autocast
     (+ NaN forward hooks on AF3-specific modules, gradient backward hooks)
  4. Module-level FP8 ablation (4 sub-experiments, each 100 steps)

Invoked via:
    python scripts/train/train.py scripts/train/configs/structure.yaml \\
        --diagnostic [--diagnostic-steps N]
"""
from __future__ import annotations

import csv
import functools
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

# ── Transformer Engine ───────────────────────────────────────────────────────
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

_FP8_RECIPE = DelayedScaling(fp8_format=Format.HYBRID)

# ── Boltz module types used for hooks / FP8 targeting ────────────────────────
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.modules.trunk import InputEmbedder

_MONITOR_TYPES = (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttention,       # covers both TriangleAttentionStartingNode & EndingNode
    PairWeightedAveraging,   # MSARowAttentionWithPairBias equivalent
)

DIAG_OUT = Path("outputs/diagnostics")


# ── Tiny utilities ────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return v.detach().float().item() if torch.is_tensor(v) else float(v)
    except Exception:
        return None


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    add_header = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if add_header:
            w.writeheader()
        w.writerows(rows)


# ── FP8 helpers ───────────────────────────────────────────────────────────────

def _replace_linear_with_te(module: nn.Module, _depth: int = 0) -> int:
    """Recursively replace nn.Linear → te.Linear in-place.

    Uses ``type(child) is nn.Linear`` (exact match) to avoid replacing
    subclasses such as transformer_engine's own Linear.

    Returns the number of replacements made.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            try:
                te_lin = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                ).to(child.weight.device)
                te_lin.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    te_lin.bias.data.copy_(child.bias.data)
                setattr(module, name, te_lin)
                replaced += 1
            except Exception as exc:
                if _depth == 0:
                    print(f"    [warn] te.Linear swap failed for {name}: {exc}")
        else:
            replaced += _replace_linear_with_te(child, _depth + 1)
    return replaced


def _wrap_fp8_autocast(model: nn.Module, target_types: tuple) -> list:
    """Wrap the ``forward`` of every module matching *target_types* with
    ``fp8_autocast``.  Returns a restore list of ``(module, original_forward)``
    so callers can undo the patch.
    """
    handles: list[tuple] = []
    recipe = _FP8_RECIPE
    for mod in model.modules():
        if isinstance(mod, target_types):
            orig = mod.forward

            @functools.wraps(orig)
            def _fwd(*args, _o=orig, _r=recipe, **kwargs):
                with te.fp8_autocast(enabled=True, fp8_recipe=_r):
                    return _o(*args, **kwargs)

            mod.forward = _fwd
            handles.append((mod, orig))
    return handles


def _restore_fp8_wraps(handles: list) -> None:
    for mod, orig in handles:
        mod.forward = orig


# ── Diagnostic PL Callback ────────────────────────────────────────────────────

class DiagnosticCallback(pl.Callback):
    """Collects per-step metrics, detects NaN, and tracks gradient statistics.

    Parameters
    ----------
    csv_path:
        Path to the per-step metrics CSV.
    log_every:
        Log a row every *log_every* global steps.
    reference_losses:
        Mapping ``{step: loss}`` used to flag divergence (> 2× reference).
    enable_nan_hooks:
        Register forward hooks on AF3-specific modules to detect the first NaN.
    enable_grad_stats:
        Register per-parameter backward hooks and write to *grad_csv_path*.
    grad_csv_path:
        CSV path for gradient statistics.
    grad_log_every:
        Collect gradient stats every *grad_log_every* steps.
    """

    def __init__(
        self,
        csv_path: Path,
        log_every: int = 10,
        reference_losses: Optional[dict] = None,
        enable_nan_hooks: bool = False,
        enable_grad_stats: bool = False,
        grad_csv_path: Optional[Path] = None,
        grad_log_every: int = 50,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.log_every = log_every
        self.reference_losses: dict = reference_losses or {}
        self.enable_nan_hooks = enable_nan_hooks
        self.enable_grad_stats = enable_grad_stats
        self.grad_csv_path = grad_csv_path
        self.grad_log_every = grad_log_every

        # mutable state shared with hooks via single-element lists
        self._step: list[int] = [0]
        self._lr: list[float] = [0.0]

        self.records: list[dict] = []
        self.grad_records: list[dict] = []
        self._nan_first: Optional[str] = None
        self.diverge_step: Optional[int] = None

        self._step_start: float = 0.0
        self._fwd_handles: list = []
        self._bwd_handles: list = []

    # ── PL lifecycle ──────────────────────────────────────────────────────────

    def setup(self, trainer, pl_module, stage: str) -> None:
        if self.enable_nan_hooks:
            self._attach_nan_hooks(pl_module)
        if self.enable_grad_stats:
            self._attach_grad_hooks(pl_module)

    def teardown(self, trainer, pl_module, stage: str) -> None:
        for h in self._fwd_handles:
            h.remove()
        self._fwd_handles.clear()
        for h in self._bwd_handles:
            h.remove()
        self._bwd_handles.clear()
        # flush remaining grad records
        if self.grad_records and self.grad_csv_path:
            _append_csv(self.grad_csv_path, self.grad_records)
            self.grad_records.clear()

    # ── per-batch hooks ───────────────────────────────────────────────────────

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._step_start = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        step = trainer.global_step
        self._step[0] = step
        elapsed_ms = (time.perf_counter() - self._step_start) * 1000.0

        try:
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            peak_mb = 0.0

        try:
            self._lr[0] = trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            pass

        # extract scalar loss from whatever training_step returned
        loss: Optional[float] = None
        if outputs is not None:
            if torch.is_tensor(outputs):
                loss = _safe_float(outputs)
            elif isinstance(outputs, dict):
                loss = _safe_float(outputs.get("loss"))

        # divergence detection
        diverged = False
        if loss is not None and self.reference_losses:
            ref = self.reference_losses.get(step)
            if ref is not None and ref > 0.0 and loss > 2.0 * ref:
                diverged = True
                if self.diverge_step is None:
                    self.diverge_step = step

        if step % self.log_every == 0:
            rec = {
                "step": step,
                "loss": loss,
                "peak_memory_mb": round(peak_mb, 1),
                "step_time_ms": round(elapsed_ms, 1),
                "throughput_steps_per_sec": round(
                    1000.0 / max(elapsed_ms, 1e-9), 4
                ),
                "diverged": diverged,
                "nan_module": self._nan_first,
            }
            self.records.append(rec)
            _append_csv(self.csv_path, [rec])

    # ── NaN forward hooks ─────────────────────────────────────────────────────

    def _attach_nan_hooks(self, model: nn.Module) -> None:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, _MONITOR_TYPES):
                h = mod.register_forward_hook(self._make_nan_hook(mod_name))
                self._fwd_handles.append(h)

    def _make_nan_hook(self, name: str):
        def _hook(mod, inp, out):
            if self._nan_first is not None:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and (
                torch.isnan(t).any() or torch.isinf(t).any()
            ):
                self._nan_first = name

        return _hook

    # ── gradient backward hooks ───────────────────────────────────────────────

    def _attach_grad_hooks(self, model: nn.Module) -> None:
        """Register per-parameter hooks on:
        - All TriangleMult / TriangleAttention / PairWeightedAveraging params
        - Up to 5 params from any InputEmbedder (to sample standard linears)
        """
        embedder_count = 0
        for mod_name, mod in model.named_modules():
            is_embedder = isinstance(mod, InputEmbedder)
            should_monitor = isinstance(mod, _MONITOR_TYPES) or is_embedder
            if not should_monitor:
                continue
            for p_name, param in mod.named_parameters(recurse=False):
                if is_embedder and embedder_count >= 5:
                    break
                full_name = f"{mod_name}.{p_name}"
                h = param.register_hook(self._make_grad_hook(full_name))
                self._bwd_handles.append(h)
                if is_embedder:
                    embedder_count += 1

    def _make_grad_hook(self, name: str):
        step_ref = self._step
        lr_ref = self._lr
        every = self.grad_log_every

        def _hook(grad: torch.Tensor) -> None:
            if step_ref[0] % every != 0:
                return
            try:
                max_abs = grad.abs().max().item()
                mean_abs = grad.abs().mean().item()
                has_nan = bool(torch.isnan(grad).any())
                rec = {
                    "step": step_ref[0],
                    "module": name,
                    "max_abs_grad": round(max_abs, 8),
                    "mean_abs_grad": round(mean_abs, 8),
                    "has_nan": has_nan,
                    "update_exceeds_lr_bound": bool(max_abs > lr_ref[0] * 1.5),
                }
                self.grad_records.append(rec)
            except Exception:
                pass

        return _hook

    # ── summary accessors ─────────────────────────────────────────────────────

    @property
    def final_loss(self) -> Optional[float]:
        for rec in reversed(self.records):
            if rec["loss"] is not None:
                return rec["loss"]
        return None

    @property
    def losses_by_step(self) -> dict:
        return {
            rec["step"]: rec["loss"]
            for rec in self.records
            if rec["loss"] is not None
        }

    @property
    def peak_memory_mb(self) -> float:
        if not self.records:
            return 0.0
        return max(rec["peak_memory_mb"] for rec in self.records)


# ── Single experiment runner ───────────────────────────────────────────────────

@dataclass
class _ExpResult:
    label: str
    final_loss: Optional[float]
    diverge_step: Optional[int]
    peak_memory_mb: float
    nan_module: Optional[str]


def _run_one(
    raw_config_path: str,
    base_args: list[str],
    config_overrides: dict,
    callback: DiagnosticCallback,
    fp8_mode: Optional[str] = None,
    seed: int = 42,
    label: str = "",
) -> None:
    """Instantiate model + data from config, run PL trainer, then clean up.

    Parameters
    ----------
    fp8_mode:
        ``None``           – no FP8
        ``"all"``          – replace nn.Linear with te.Linear everywhere
        ``"msa"``          – fp8_autocast on PairWeightedAveraging
        ``"tri_attn"``     – fp8_autocast on TriangleAttention
        ``"tri_mult"``     – fp8_autocast on TriangleMult{Outgoing,Incoming}
        ``"input_embed"``  – fp8_autocast on InputEmbedder
    """
    import hydra
    import omegaconf
    from omegaconf import OmegaConf
    from boltz.data.module.training import BoltzTrainingDataModule, DataConfig
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True

    # ── Build merged config ───────────────────────────────────────────────────
    raw = omegaconf.OmegaConf.load(raw_config_path)
    all_args = list(base_args) + [f"{k}={v}" for k, v in config_overrides.items()]
    raw = omegaconf.OmegaConf.merge(raw, omegaconf.OmegaConf.from_dotlist(all_args))

    # Force diagnostic-friendly settings before instantiation
    OmegaConf.update(raw, "data.num_workers", 0, merge=True)
    OmegaConf.update(raw, "trainer.devices", 1, merge=True)

    try:
        cfg = hydra.utils.instantiate(raw)
    except Exception as exc:
        print(f"  [error] Config instantiation failed: {exc}")
        return

    model = cfg.model
    trainer_kw = dict(cfg.trainer) if cfg.trainer else {}
    # Ensure debug-mode keys are applied
    trainer_kw["devices"] = 1

    data_config = DataConfig(**cfg.data)
    data_module = BoltzTrainingDataModule(data_config)

    # ── FP8 patching ─────────────────────────────────────────────────────────
    fp8_restore: list = []
    if fp8_mode:
        if fp8_mode == "all":
            n = _replace_linear_with_te(model)
            print(f"  Replaced {n} nn.Linear → te.Linear")
            # Also wrap any remaining (e.g. inside activation-checkpointed blocks)
            fp8_restore = _wrap_fp8_autocast(model, (nn.Linear,))
        else:
            target_map: dict = {
                "msa":         (PairWeightedAveraging,),
                "tri_attn":    (TriangleAttention,),
                "tri_mult":    (TriangleMultiplicationOutgoing,
                                TriangleMultiplicationIncoming),
                "input_embed": (InputEmbedder,),
            }
            targets = target_map.get(fp8_mode, ())
            if targets:
                fp8_restore = _wrap_fp8_autocast(model, targets)
                print(f"  fp8_autocast applied to {len(fp8_restore)} {fp8_mode} modules")

    # ── Run trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        default_root_dir=str(cfg.output),
        callbacks=[callback],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=1,
        **trainer_kw,
    )

    print(f"  Running {label} …")
    try:
        trainer.fit(model, datamodule=data_module)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  [OOM] {exc}")
        _append_csv(
            callback.csv_path,
            [{"step": "OOM", "loss": None, "peak_memory_mb": None,
              "step_time_ms": None, "throughput_steps_per_sec": None,
              "diverged": False, "nan_module": None}],
        )
    except Exception as exc:
        print(f"  [error during fit] {type(exc).__name__}: {exc}")
    finally:
        _restore_fp8_wraps(fp8_restore)
        del trainer, model, data_module
        gc.collect()
        torch.cuda.empty_cache()


# ── Subprocess worker (module-level so it is picklable for spawn) ────────────

def _exp_worker(
    result_q,
    raw_config_path: str,
    base_args: list,
    config_overrides: dict,
    csv_path: str,
    log_every: int,
    reference_losses: dict,
    enable_nan_hooks: bool,
    enable_grad_stats: bool,
    grad_csv_path,
    grad_log_every: int,
    fp8_mode,
    seed: int,
    label: str,
) -> None:
    """Runs inside a spawned subprocess for a clean CUDA allocator state."""
    cb = DiagnosticCallback(
        csv_path=Path(csv_path),
        log_every=log_every,
        reference_losses=reference_losses,
        enable_nan_hooks=enable_nan_hooks,
        enable_grad_stats=enable_grad_stats,
        grad_csv_path=Path(grad_csv_path) if grad_csv_path else None,
        grad_log_every=grad_log_every,
    )
    _run_one(raw_config_path, base_args, config_overrides, cb, fp8_mode, seed, label)
    result_q.put({
        "final_loss": cb.final_loss,
        "diverge_step": cb.diverge_step,
        "peak_memory_mb": cb.peak_memory_mb,
        "nan_first": cb._nan_first,
        "losses_by_step": cb.losses_by_step,
    })


def _run_subprocess(
    raw_config_path: str,
    base_args: list,
    config_overrides: dict,
    csv_path: Path,
    fp8_mode=None,
    seed: int = 42,
    label: str = "",
    log_every: int = 10,
    reference_losses: Optional[dict] = None,
    enable_nan_hooks: bool = False,
    enable_grad_stats: bool = False,
    grad_csv_path: Optional[Path] = None,
    grad_log_every: int = 50,
) -> dict:
    """Spawn a fresh process for one experiment and return its result dict.

    Each experiment runs in an isolated CUDA context so peak_memory_mb
    reflects only that experiment's allocations, with no fragmentation
    carried over from previous runs.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_exp_worker,
        args=(
            q, raw_config_path, base_args, config_overrides,
            str(csv_path), log_every, reference_losses or {},
            enable_nan_hooks, enable_grad_stats,
            str(grad_csv_path) if grad_csv_path else None, grad_log_every,
            fp8_mode, seed, label,
        ),
    )
    p.start()
    p.join()
    if not q.empty():
        return q.get()
    return {
        "final_loss": None,
        "diverge_step": None,
        "peak_memory_mb": 0.0,
        "nan_first": None,
        "losses_by_step": {},
    }


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_all_diagnostics(
    raw_config_path: str,
    base_args: list[str],
    n_steps: int = 200,
) -> None:
    """Run all diagnostic experiments and print a summary table.

    Parameters
    ----------
    raw_config_path : str
        Path to the YAML training config.
    base_args : list[str]
        Remaining CLI overrides passed through from train.py.
    n_steps : int
        Number of training steps for experiments 1-3 (exp 4 uses n_steps//2).
    """
    DIAG_OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  FP8 Diagnostics  |  output: {DIAG_OUT}  |  steps={n_steps}")
    print(f"{'='*72}\n")

    results: list[_ExpResult] = []
    common: dict = {
        "trainer.accumulate_grad_batches": 1,
        "trainer.limit_train_batches": n_steps,
        "trainer.limit_val_batches": 16,
        "debug": 1,
    }

    # ── Experiment 1: FP32 baseline ──────────────────────────────────────────
    print(f"[1/4] FP32 Baseline  ({n_steps} steps)")
    r1 = _run_subprocess(
        raw_config_path, base_args,
        {**common, "trainer.precision": 32, "trainer.max_steps": n_steps},
        csv_path=DIAG_OUT / "baseline_fp32.csv",
        seed=42, label="Exp1-FP32",
    )
    fp32_losses = r1["losses_by_step"]
    results.append(_ExpResult(
        label="1  FP32 Baseline",
        final_loss=r1["final_loss"],
        diverge_step=None,
        peak_memory_mb=r1["peak_memory_mb"],
        nan_module=r1["nan_first"],
    ))
    print(f"  → loss={r1['final_loss']}  mem={r1['peak_memory_mb']:.0f} MB\n")

    # ── Experiment 2: BF16 baseline ───────────────────────────────────────────
    print(f"[2/4] BF16-mixed Baseline  ({n_steps} steps)")
    r2 = _run_subprocess(
        raw_config_path, base_args,
        {**common, "trainer.precision": "bf16-mixed", "trainer.max_steps": n_steps},
        csv_path=DIAG_OUT / "baseline_bf16.csv",
        seed=42, label="Exp2-BF16",
        reference_losses=fp32_losses,
    )
    bf16_losses = r2["losses_by_step"]
    results.append(_ExpResult(
        label="2  BF16 Baseline",
        final_loss=r2["final_loss"],
        diverge_step=r2["diverge_step"],
        peak_memory_mb=r2["peak_memory_mb"],
        nan_module=r2["nan_first"],
    ))
    print(
        f"  → loss={r2['final_loss']}  mem={r2['peak_memory_mb']:.0f} MB"
        f"  diverge_step={r2['diverge_step'] or '—'}\n"
    )

    # ── Experiment 3: Naive FP8 (all linears) ────────────────────────────────
    print(f"[3/4] Naive FP8 – all Linear  ({n_steps} steps)")
    r3 = _run_subprocess(
        raw_config_path, base_args,
        {**common, "trainer.precision": "bf16-mixed", "trainer.max_steps": n_steps},
        csv_path=DIAG_OUT / "naive_fp8_all.csv",
        fp8_mode="all", seed=42, label="Exp3-FP8-all",
        reference_losses=bf16_losses,
        enable_nan_hooks=True,
        enable_grad_stats=True,
        grad_csv_path=DIAG_OUT / "gradient_stats.csv",
        grad_log_every=50,
    )
    results.append(_ExpResult(
        label="3  FP8 All Linear",
        final_loss=r3["final_loss"],
        diverge_step=r3["diverge_step"],
        peak_memory_mb=r3["peak_memory_mb"],
        nan_module=r3["nan_first"],
    ))
    print(
        f"  → loss={r3['final_loss']}  mem={r3['peak_memory_mb']:.0f} MB"
        f"  diverge_step={r3['diverge_step'] or '—'}"
        f"  nan_module={r3['nan_first'] or '—'}\n"
    )

    # ── Experiment 4: Module-level FP8 ablation ───────────────────────────────
    sub_steps = max(n_steps // 2, 50)
    print(f"[4/4] Module-level FP8 Ablation  ({sub_steps} steps × 4 sub-experiments)")
    sub_experiments = [
        ("4a", "input_embed", "InputEmbedder (standard linear only)"),
        ("4b", "msa",         "PairWeightedAveraging (MSARowAttn equiv)"),
        ("4c", "tri_attn",    "TriangleAttention (start + end node)"),
        ("4d", "tri_mult",    "TriangleMult (outgoing + incoming)"),
    ]
    ablation_rows: list[dict] = []

    for sub_id, fp8_mode, desc in sub_experiments:
        print(f"  [{sub_id}] FP8 on: {desc}")
        r4 = _run_subprocess(
            raw_config_path, base_args,
            {**common, "trainer.precision": "bf16-mixed",
             "trainer.max_steps": sub_steps},
            csv_path=DIAG_OUT / f"ablation_{sub_id}_{fp8_mode}.csv",
            fp8_mode=fp8_mode, seed=42, label=f"Exp{sub_id}",
            reference_losses=bf16_losses,
        )
        ablation_rows.append({
            "sub_experiment": sub_id,
            "fp8_module_group": desc,
            "final_loss": r4["final_loss"],
            "divergence_occurred": r4["diverge_step"] is not None,
            "first_divergence_step": r4["diverge_step"],
            "peak_memory_mb": r4["peak_memory_mb"],
        })
        results.append(_ExpResult(
            label=f"4{sub_id} FP8-{fp8_mode}",
            final_loss=r4["final_loss"],
            diverge_step=r4["diverge_step"],
            peak_memory_mb=r4["peak_memory_mb"],
            nan_module=r4["nan_first"],
        ))
        print(
            f"    → loss={r4['final_loss']}"
            f"  diverge_step={r4['diverge_step'] or '—'}\n"
        )
    _append_csv(DIAG_OUT / "module_ablation.csv", ablation_rows)

    # ── Summary table ─────────────────────────────────────────────────────────
    cols = [
        ("Experiment",      28),
        ("Final Loss",      14),
        ("Diverge Step",    14),
        ("Peak Memory MB",  16),
        ("NaN Module",      30),
    ]

    header = "  ".join(name.ljust(w) for name, w in cols)
    sep    = "  ".join("-" * w for _, w in cols)

    print(f"\n{'='*72}")
    print("  RESULTS SUMMARY")
    print(f"{'='*72}")
    print(header)
    print(sep)

    for r in results:
        row_vals = [
            r.label,
            r.final_loss,
            r.diverge_step or "—",
            r.peak_memory_mb,
            r.nan_module or "—",
        ]
        line = "  ".join(
            (str(v)[:w]).ljust(w) if not isinstance(v, float)
            else f"{v:.4f}".ljust(w)
            for v, (_, w) in zip(row_vals, cols)
        )
        print(line)

    print(f"\n  CSVs written to: {DIAG_OUT.resolve()}")
    print(f"{'='*72}\n")
