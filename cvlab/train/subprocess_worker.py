"""Training subprocess worker — launched by `cvlab train` as subprocess.

Usage:
    python -m cvlab.train.subprocess_worker \\
        --experiment-id exp_001 \\
        --config /path/to/config.yaml \\
        --batch-size 64

Exit codes:
    0   Success
    1   Training failure (non-OOM exception)
    137 CUDA Out of Memory (includes both RuntimeError and system OOM)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    try:
        from cvlab.train.run import train_classification

        train_classification(
            config_path=args.config,
            experiment_id=args.experiment_id,
            batch_size=args.batch_size,
        )
        return 0

    except Exception as e:
        import torch

        err_msg = str(e).lower()

        # CUDA OOM detection — covers multiple paths:
        # 1. torch.cuda.OutOfMemoryError (explicit)
        # 2. RuntimeError with "CUDA out of memory" in message (most common)
        # 3. System OOM killer (SIGKILL, exit code 137 from OS)
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "cuda out of memory" in err_msg

        if is_oom:
            sys.stderr.write("CUDA out of memory\n")
            return 137
        else:
            sys.stderr.write(f"Training failed: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
