from pathlib import Path

import hydra
from hydra import instantiate

from src.metrics.utils import calc_cer, calc_wer


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    """
    Calculate CER and WER given dirs with ground truth
    and predicted transcriptions. Files are matched by name.

    Args:
        config (DictConfig): hydra config with gt_dir and pred_dir.
    """
    text_encoder = instantiate(config.text_encoder)
    target_dir, pred_dir = Path(config.target_dir), Path(config.pred_dir)

    cers, wers = [], []
    for target_path in sorted(target_dir.glob("*.txt")):
        pred_path = pred_dir / target_path.name
        if not pred_path.exists():
            print(f"No pred for {target_path.stem}")
            continue

        target = text_encoder.normalize_text(target_path.read_text().strip())
        pred = text_encoder.normalize_text(pred_path.read_text().strip())
        cers.append(calc_cer(target, pred))
        wers.append(calc_wer(target, pred))

    print(f"CER - {sum(cers) / len(cers)}")
    print(f"WER - {sum(wers) / len(wers)}")


if __name__ == "__main__":
    main()
