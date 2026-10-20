import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)

    args = parser.parse_args()

    text_encoder = CTCTextEncoder()

    cers, wers = [], []
    for target_path in sorted(args.target_dir.glob("*.txt")):
        pred_path = args.pred_dir / target_path.name
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
