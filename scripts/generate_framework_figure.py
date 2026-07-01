from __future__ import annotations

import argparse

from eeg_cogagent.config import load_config, project_path
from eeg_cogagent.viz import plot_workflow_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the EEG-CogAgent workflow figure.")
    parser.add_argument("--config", default="configs/ds004504_minimal.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = project_path(cfg, cfg["paths"]["output_dir"]) / "figures"
    paths = plot_workflow_figure(output_dir)
    print("Generated: " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
