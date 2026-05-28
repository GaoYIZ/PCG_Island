"""
Import an RL-exported island package into the Unity/Tuanjie terrain project.

Typical use after training:
    python import_rl_export_to_unity.py --rl-output-dir formal_rl_from_3k_v3_voronoi_drop_connectivity_tuned_v2 --agent ppo --rank 1 --open-unity

You can also point directly at an exported island_config.json:
    python import_rl_export_to_unity.py --config path/to/island_config.json --open-unity
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_UNITY_EXE = Path(r"E:\Unity\Editor\Tuanjie.exe")
DEFAULT_UNITY_PROJECT = Path(r"E:\IslandTest\unity\PCGIslandUnity")
EXECUTE_METHOD = "PCGIslandTerrainGeneratorWindow.ImportCurrentIslandFromCommandLine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Unity-ready RL island export.")
    parser.add_argument("--config", type=Path, default=None, help="Direct path to island_config.json")
    parser.add_argument(
        "--rl-output-dir",
        type=Path,
        default=Path("formal_rl_from_3k_v3_voronoi_drop_connectivity_tuned_v2"),
        help="Formal RL output directory containing ppo_unity_exports/sac_unity_exports.",
    )
    parser.add_argument("--agent", choices=("ppo", "sac"), default="ppo", help="Agent export folder to import")
    parser.add_argument("--rank", type=int, default=1, help="Top-N export rank to import")
    parser.add_argument("--unity-exe", type=Path, default=DEFAULT_UNITY_EXE)
    parser.add_argument("--unity-project", type=Path, default=DEFAULT_UNITY_PROJECT)
    parser.add_argument("--screenshot-width", type=int, default=1600)
    parser.add_argument("--screenshot-height", type=int, default=1000)
    parser.add_argument("--open-unity", action="store_true", help="Open the Unity project after batch import")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Path:
    if args.config is not None:
        config_path = args.config.resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    export_root = (args.rl_output_dir / f"{args.agent.lower()}_unity_exports").resolve()
    if not export_root.exists():
        raise FileNotFoundError(f"Unity export folder not found: {export_root}")

    candidates = sorted(path for path in export_root.glob("top_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No top_* exports found in: {export_root}")
    if args.rank < 1 or args.rank > len(candidates):
        raise ValueError(f"--rank must be in [1, {len(candidates)}], got {args.rank}")

    config_path = candidates[args.rank - 1] / "island_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path.resolve()


def run_unity_import(args: argparse.Namespace, config_path: Path) -> tuple[Path, Path]:
    unity_exe = args.unity_exe.resolve()
    unity_project = args.unity_project.resolve()
    if not unity_exe.exists():
        raise FileNotFoundError(f"Unity/Tuanjie executable not found: {unity_exe}")
    if not unity_project.exists():
        raise FileNotFoundError(f"Unity project not found: {unity_project}")

    export_dir = config_path.parent
    screenshot_path = export_dir / "unity_screenshot.png"
    log_path = export_dir / "unity_import.log"
    command = [
        str(unity_exe),
        "-batchmode",
        "-quit",
        "-projectPath",
        str(unity_project),
        "-executeMethod",
        EXECUTE_METHOD,
        "-pcgIslandJson",
        str(config_path),
        "-pcgIslandScreenshot",
        str(screenshot_path),
        "-pcgIslandScreenshotWidth",
        str(args.screenshot_width),
        "-pcgIslandScreenshotHeight",
        str(args.screenshot_height),
        "-logFile",
        str(log_path),
    ]
    completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parent))
    if completed.returncode != 0:
        raise RuntimeError(f"Unity import failed with exit code {completed.returncode}. See: {log_path}")
    if not screenshot_path.exists():
        raise RuntimeError(f"Unity import completed but screenshot was not created: {screenshot_path}")
    return screenshot_path, log_path


def open_unity_project(args: argparse.Namespace, config_path: Path) -> Path:
    unity_exe = args.unity_exe.resolve()
    unity_project = args.unity_project.resolve()
    log_path = config_path.parent / "unity_editor_open.log"
    subprocess.Popen(
        [
            str(unity_exe),
            "-projectPath",
            str(unity_project),
            "-logFile",
            str(log_path),
        ],
        cwd=str(Path(__file__).resolve().parent),
    )
    return log_path


def main() -> int:
    args = parse_args()
    config_path = resolve_config(args)
    screenshot_path, log_path = run_unity_import(args, config_path)
    editor_log_path = open_unity_project(args, config_path) if args.open_unity else None

    print("RL island imported into Unity.")
    print(f"  config: {config_path}")
    print(f"  screenshot: {screenshot_path}")
    print(f"  import_log: {log_path}")
    if editor_log_path is not None:
        print(f"  editor_log: {editor_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
