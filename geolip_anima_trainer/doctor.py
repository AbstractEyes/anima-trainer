#!/usr/bin/env python3
"""
doctor.py — environment diagnostics for Anima finetuning.

Runs on both the local RTX 4090 (sm_89, smoke-test) box and the Blackwell sm_120
target. Every check is wrapped so a missing dependency yields WARN/INFO, never a
traceback. The target rules are baked in: bf16 YES, fp8 NO, flash-attn NO. The one
hard FAIL is the dangerous case: an sm_120 device with a torch whose CUDA build is
< 12.8 (that torch cannot emit Blackwell kernels).
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    def render(self) -> str:
        lines = ["== Anima Trainer Doctor =="]
        for c in self.checks:
            lines.append(f"[{c.status}] {c.name:<24} {c.detail}")
        n_fail = sum(c.status == FAIL for c in self.checks)
        n_warn = sum(c.status == WARN for c in self.checks)
        lines.append("----")
        verdict = "READY" if self.ok else "NOT READY"
        lines.append(f"SUMMARY: {n_fail} FAIL, {n_warn} WARN  -> {verdict}")
        lines.append("Target rules: bf16 OK | fp8 disabled | flash-attn absent OK")
        return "\n".join(lines)


def _is_windows() -> bool:
    return platform.system() == "Windows"


def doctor(*, repo_root: str | Path | None = None,
           config_toml: str | Path | None = None) -> DoctorReport:
    """Build a structured environment report. Pure read-only; never mutates."""
    rep = DoctorReport()

    # 1. OS / arch
    os_note = "  (training disabled here - smoke-test box)" if _is_windows() else ""
    rep.add("os_arch", INFO, f"{platform.system()} {platform.machine()}{os_note}")

    # 2. Python
    v = sys.version_info
    rep.add("python_version", PASS if (3, 12) <= (v.major, v.minor) else WARN,
            f"{v.major}.{v.minor}.{v.micro}" +
            ("" if (3, 12) <= (v.major, v.minor) else "  (3.12 expected)"))

    # 3-11. torch-dependent
    torch = None
    try:
        import torch as _torch
        torch = _torch
        rep.add("torch_installed", PASS, torch.__version__)
    except Exception as e:  # noqa: BLE001
        rep.add("torch_installed", FAIL,
                f"not importable ({e.__class__.__name__}) — install per smoke-test step 2")

    sm = None
    cuda_build = None
    if torch is not None:
        cuda_build = torch.version.cuda
        rep.add("torch_cuda_build", PASS if cuda_build else WARN,
                cuda_build or "None (CPU-only wheel)")
        try:
            avail = torch.cuda.is_available()
            rep.add("cuda_available", PASS if avail else WARN, str(avail))
            if avail:
                rep.add("device_name", INFO, torch.cuda.get_device_name(0))
                cap = torch.cuda.get_device_capability(0)
                sm = cap[0] * 10 + cap[1]
                rep.add("compute_capability", INFO, f"sm_{sm}  (cap {cap})")
                rep.add("bf16_supported", PASS if torch.cuda.is_bf16_supported() else FAIL,
                        str(torch.cuda.is_bf16_supported()))
        except Exception as e:  # noqa: BLE001
            rep.add("cuda_available", WARN, f"probe failed: {e}")
        rep.add("sdpa_available",
                PASS if hasattr(torch.nn.functional, "scaled_dot_product_attention") else FAIL,
                "scaled_dot_product_attention present" if
                hasattr(torch.nn.functional, "scaled_dot_product_attention") else "missing")

    # The single most important target gate.
    if sm is not None and sm >= 120:
        def _cuda_ge_128(cb: str | None) -> bool:
            if not cb:
                return False
            try:
                maj, _, mnr = cb.partition(".")
                return (int(maj), int(mnr or 0)) >= (12, 8)
            except ValueError:
                return False
        rep.add("blackwell_torch", PASS if _cuda_ge_128(cuda_build) else FAIL,
                f"sm_{sm} needs torch cuda >= 12.8 (cu128); got {cuda_build}")

    # 9. fp8 policy (standing guardrail)
    rep.add("fp8_policy", INFO,
            "fp8 disabled by design (e4m3 degrades this lineage; 96GB makes it pointless)")

    # 10. flash-attn must be ABSENT
    if importlib.util.find_spec("flash_attn") is not None:
        rep.add("flash_attn_absent", WARN,
                "flash-attn IS installed — broken on sm_120; `pip uninstall flash-attn` (SDPA is correct)")
    else:
        rep.add("flash_attn_absent", PASS, "not installed (correct - SDPA backend)")

    # 12. deepspeed
    ds_present = importlib.util.find_spec("deepspeed") is not None
    if _is_windows():
        rep.add("deepspeed_import", WARN if not ds_present else INFO,
                "not importable; OK on Windows (Linux-only)" if not ds_present
                else "present (note: training still won't run on Windows)")
    else:
        rep.add("deepspeed_import", PASS if ds_present else FAIL,
                "present" if ds_present else "missing — install diffusion-pipe requirements")

    # 13. diffusion-pipe source
    try:
        from .launch import find_diffusion_pipe
        train_py = find_diffusion_pipe(repo_root)
        rep.add("diffusion_pipe", PASS, str(train_py))
    except Exception as e:  # noqa: BLE001
        rep.add("diffusion_pipe", FAIL, str(e).splitlines()[0])

    # 16. config sanity + model files (if a config was supplied)
    if config_toml is not None and Path(config_toml).is_file():
        try:
            t = tomllib.loads(Path(config_toml).read_text(encoding="utf-8"))
            model = t.get("model", {})
            llm_lr = model.get("llm_adapter_lr", 0)
            notes = (f"type={model.get('type')} dtype={model.get('dtype')} "
                     f"llm_adapter_lr={llm_lr} "
                     f"activation_checkpointing={t.get('activation_checkpointing')}")
            status = PASS
            if model.get("dtype") != "bfloat16":
                status = WARN
            if isinstance(llm_lr, (int, float)) and llm_lr > 5e-6:
                status = WARN
            rep.add("config_sanity", status, notes)

            # Model files; absolute Linux paths on Windows are not checkable -> INFO.
            for key in ("transformer_path", "vae_path", "llm_path"):
                p = model.get(key, "")
                if not p:
                    rep.add(f"model.{key}", WARN, "empty")
                elif _is_windows() and p.startswith("/"):
                    rep.add(f"model.{key}", INFO, f"target path (not checkable here): {p}")
                else:
                    rep.add(f"model.{key}", PASS if Path(p).exists() else WARN,
                            p if Path(p).exists() else f"missing: {p}")
        except Exception as e:  # noqa: BLE001
            rep.add("config_sanity", WARN, f"could not parse {config_toml}: {e}")

    return rep
