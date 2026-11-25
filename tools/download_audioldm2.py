import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "cvssp/audioldm2"
MODEL_REVISION = "c8e7e189d324425c05c4c2f81214041ef4107983"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pinned AudioLDM 2 safetensors snapshot."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "audio" / "audioldm2",
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(output_dir),
        allow_patterns=("*.safetensors", "*.json", "*.txt", "*.model"),
    )
    print(f"Downloaded {MODEL_ID}@{MODEL_REVISION} to {output_dir}")


if __name__ == "__main__":
    main()
