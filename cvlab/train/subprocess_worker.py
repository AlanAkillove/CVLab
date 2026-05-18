"""训练子进程 Worker — 由 cvlab train 以 subprocess 方式启动。

用法:
    python -m cvlab.train.subprocess_worker \\
        --experiment-id exp_001 \\
        --config /path/to/config.yaml \\
        --batch-size 64

退出码:
    0  成功
    1  训练失败（非 OOM 异常）
    137 CUDA Out of Memory
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
        if isinstance(e, torch.cuda.OutOfMemoryError):
            # CUDA OOM — 退出码 137 对应 SIGKILL
            sys.stderr.write("CUDA out of memory\n")
            return 137
        sys.stderr.write(f"Training failed: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
