#!/bin/bash
# Run on the Linux cluster, from the repository root. Isolate the runtime
# from the locked Python environment and retain it on shared storage.
set -euo pipefail
prefix="$PWD/.ca-runtime"
if [[ ! -s "$prefix/lib/libtbb.so.12" ]]; then
  uv pip install --python .venv/bin/python --prefix "$prefix" --no-deps 'tbb==2022.3.0'
fi
test -s "$prefix/lib/libtbb.so.12"
LD_LIBRARY_PATH="$prefix/lib:$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .venv/bin/python - <<'PYTHON'
import ctypes
import sysconfig
from pathlib import Path

# Resolve the compiled extension's native dependencies without importing
# GreedTok, Transformers, or PyTorch on the login node.
extension = Path(sysconfig.get_path("platlib")) / "pcatt" / (
    "greedy_encoder" + sysconfig.get_config_var("EXT_SUFFIX")
)
ctypes.CDLL(str(extension))
print("PCATT native extension loads. Run the GPU pilot for the full training check.")
PYTHON
