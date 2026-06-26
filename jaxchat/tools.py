"""Lightweight local tool execution helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def execute_python(code: str, *, timeout_s: float = 2.0) -> dict:
    """Execute Python code in a short-lived subprocess with a restricted env.

    This is a convenience sandbox, not a security boundary.
    """
    wrapped = (
        "import json\n"
        "ns = {}\n"
        f"code = {code!r}\n"
        "try:\n"
        "    try:\n"
        "        result = eval(code, {'__builtins__': __builtins__}, ns)\n"
        "        if result is not None:\n"
        "            print(result)\n"
        "    except SyntaxError:\n"
        "        exec(code, {'__builtins__': __builtins__}, ns)\n"
        "except Exception as e:\n"
        "    print(type(e).__name__ + ': ' + str(e))\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
    }
    with tempfile.TemporaryDirectory(prefix="jaxchat-tool-") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", wrapped],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            output = stdout if stdout else stderr
            return {"ok": proc.returncode == 0, "output": output[:4000], "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": f"TimeoutExpired: exceeded {timeout_s:.1f}s", "returncode": -1}
