"""Optional Weights & Biases integration for jaxchat training scripts.

Usage::

    from jaxchat.wandb_log import wandb_init, wandb_log, wandb_finish

    wandb_init(stage="base", run_name="d4-base", config_dict={...})
    wandb_log({"step": step, "loss": loss})
    wandb_finish()

If ``WANDB_API_KEY`` is not in the environment, ``wandb`` is not installed, or
``WANDB_DISABLED=true`` is set, every call becomes a silent no-op.  This lets
the same training scripts run locally and on the cluster without code changes.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any

_RUN = None  # type: ignore[var-annotated]
_DISABLED = False


def _enabled() -> bool:
    if _DISABLED:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    # Either an env var or a credentials file is enough; wandb itself reads
    # ~/.netrc when WANDB_API_KEY is unset.
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc):
        try:
            with open(netrc, "r", encoding="utf-8") as handle:
                if "api.wandb.ai" in handle.read():
                    return True
        except OSError:
            pass
    return False


def wandb_init(*, stage: str, run_name: str | None = None, config_dict: dict | None = None) -> bool:
    """Initialize a wandb run if credentials and the SDK are available."""

    global _RUN, _DISABLED
    if _RUN is not None:
        return True
    if not _enabled():
        _DISABLED = True
        return False
    try:
        import wandb

        project = os.environ.get("WANDB_PROJECT", "jaxchat")
        entity = os.environ.get("WANDB_ENTITY") or None
        name = run_name or f"{stage}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        _RUN = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=stage,
            job_type=stage,
            config=config_dict or {},
            reinit=True,
        )
        return True
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[wandb_log] init failed ({exc!r}); disabling.")
        _DISABLED = True
        _RUN = None
        return False


def wandb_log(metrics: dict[str, Any]) -> None:
    """Forward a metrics dict to wandb if a run is active.

    Non-numeric values (e.g. ``datetime``) are stripped because wandb only
    plots scalars.  ``step`` is used as the x-axis when present.
    """

    if _RUN is None:
        return
    try:
        clean: dict[str, Any] = {}
        step = metrics.get("step")
        for key, value in metrics.items():
            if key == "step":
                continue
            if isinstance(value, (int, float, bool)):
                clean[key] = float(value)
            else:
                # Booleans are int subclasses; everything else (str/datetime)
                # is dropped to keep the dashboard tidy.
                continue
        if not clean:
            return
        if step is not None:
            _RUN.log(clean, step=int(step))
        else:
            _RUN.log(clean)
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[wandb_log] log failed ({exc!r}); disabling.")
        _disable()


def wandb_finish() -> None:
    global _RUN
    if _RUN is None:
        return
    try:
        _RUN.finish()
    except Exception:  # pragma: no cover
        pass
    _RUN = None


def _disable() -> None:
    global _RUN, _DISABLED
    _DISABLED = True
    if _RUN is not None:
        try:
            _RUN.finish()
        except Exception:  # pragma: no cover
            pass
    _RUN = None


def config_from_dataclass(config: Any, extra: dict | None = None) -> dict:
    """Best-effort dataclass → flat dict conversion for wandb config."""

    out: dict[str, Any] = {}
    fields = getattr(config, "__dataclass_fields__", None)
    if fields:
        for name in fields:
            value = getattr(config, name, None)
            if isinstance(value, (int, float, str, bool)) or value is None:
                out[name] = value
            elif isinstance(value, (tuple, list)):
                out[name] = list(value)
            else:
                out[name] = repr(value)
    if extra:
        out.update(extra)
    return out


__all__ = [
    "wandb_init",
    "wandb_log",
    "wandb_finish",
    "config_from_dataclass",
]
