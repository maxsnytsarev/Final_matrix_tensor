from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = Path(__file__).resolve().parent / "octave"


def octave_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def run_octave_bridge(
    *,
    runner_name: str,
    config: dict[str, Any],
    octave_env_name: str | None = "octave",
    cwd: Path | None = None,
) -> str:
    cfg_path = Path(config["config_path"])
    bridge_eval = (
        f"addpath('{octave_quote(BRIDGE_DIR)}'); "
        f"{runner_name}('{octave_quote(cfg_path)}');"
    )

    if octave_env_name:
        conda_sh = Path.home() / "anaconda3" / "etc" / "profile.d" / "conda.sh"
        cmd = [
            "bash",
            "-lc",
            (
                f"source {shlex.quote(str(conda_sh))} && "
                f"conda activate {shlex.quote(str(octave_env_name))} && "
                f"octave --quiet --eval {shlex.quote(bridge_eval)}"
            ),
        ]
    else:
        octave_bin = shutil.which("octave")
        if octave_bin is None:
            raise RuntimeError("Unable to find Octave executable on PATH")
        cmd = [octave_bin, "--quiet", "--eval", bridge_eval]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
        check=False,
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(
            "Octave baseline bridge failed.\n"
            f"Runner: {runner_name}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{output}"
        )
    return output


def run_mat_bridge(
    *,
    runner_name: str,
    input_payload: dict[str, Any],
    config_payload: dict[str, Any],
    octave_env_name: str | None = "octave",
) -> tuple[dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix=f"{runner_name}_") as tmpdir:
        tmp = Path(tmpdir)
        input_mat = tmp / "input.mat"
        output_mat = tmp / "output.mat"
        config_path = tmp / "config.json"

        savemat(input_mat, input_payload, do_compression=True)
        config = {
            **config_payload,
            "input_mat": str(input_mat),
            "output_mat": str(output_mat),
            "config_path": str(config_path),
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        octave_output = run_octave_bridge(
            runner_name=runner_name,
            config=config,
            octave_env_name=octave_env_name,
        )
        if not output_mat.exists():
            raise RuntimeError(
                "Octave baseline bridge finished without producing output MAT file.\n"
                f"Runner: {runner_name}\n"
                f"Output:\n{octave_output}"
            )
        return loadmat(output_mat, squeeze_me=False, struct_as_record=False), octave_output


def matlab_cell_to_arrays(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object)
    out: list[np.ndarray] = []
    for item in arr.ravel():
        if item is None:
            continue
        out.append(np.asarray(item, dtype=float))
    return out


def output_tail(output: str, *, max_lines: int = 20) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])
