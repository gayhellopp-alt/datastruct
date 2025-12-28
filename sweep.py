#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


EXPERIMENTS = [
    ("baseline", "Baseline（对照组）", []),
    ("no_gamma_early_stop", "关闭 gamma early-stop", ["-DABLATE_DISABLE_GAMMA_EARLY_STOP=1"]),
    ("cand_always_enqueue", "保留 early-stop，但 cand 无条件入队", ["-DABLATE_CAND_ALWAYS_ENQUEUE=1"]),
    ("float_distance", "SQ16 量化--->float", ["-DABLATE_USE_FLOAT_DISTANCE=1"]),
    ("serial_build", "并行构建 + 节点锁--->串行加入", ["-DABLATE_SERIAL_BUILD=1"]),
    ("no_post_process", "取消构图后处理（reverse-edge injection + L0 prune）", ["-DABLATE_DISABLE_POST_PROCESS=1"]),
    ("no_prefetch", "取消prefetch", ["-DABLATE_DISABLE_PREFETCH=1"]),
]

RETRY_TAGS = {
    "float_distance",
    "no_prefetch",
    "no_post_process",
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def build_and_run(tag: str, desc: str, defines: list[str], out_dir: Path) -> str:
    exe = out_dir / f"run_{tag}"
    cxxflags = ["-O2", "-std=c++17"]
    compile_cmd = ["g++", *cxxflags, *defines, "main.cpp", "MySolution_after.cpp", "-o", str(exe)]
    print(f"\n=== [{tag}] {desc} ===")
    print("Compile:", " ".join(compile_cmd))
    run(compile_cmd)
    print("Run:", exe)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.stdout


def write_csv(rows: list[str], path: Path) -> None:
    header = "tag,desc,defines,output\n"
    content = header + "\n".join(rows) + "\n"
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation sweep runner")
    parser.add_argument("--out-dir", default="build", help="输出目录")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, desc, defines in EXPERIMENTS:
        if tag not in RETRY_TAGS:
            continue
        output = build_and_run(tag, desc, defines, out_dir)
        define_str = " ".join(defines)
        safe_output = output.replace("\n", "\\n").replace("\r", "\\r")
        rows.append(f"{tag},{desc},{define_str},{safe_output}")

    csv_path = out_dir / "ablation_retry.csv"
    write_csv(rows, csv_path)
    print(f"\nCSV 输出: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
