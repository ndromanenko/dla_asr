from pathlib import Path

from huggingface_hub import hf_hub_download

SAVE_DIR = Path(__file__).absolute().resolve().parent.parent / "ckpt"


def main():
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    path = hf_hub_download(
        repo_id="romanenko02/dla-asr-conformer",
        filename="model_best.pth",
        local_dir=SAVE_DIR,
    )
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
