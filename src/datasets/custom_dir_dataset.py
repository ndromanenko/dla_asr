from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, dir, *args, **kwargs):
        data = []
        audio_dir = dir + "/audio"
        transcription_dir = dir + "/transcriptions"
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                entry["text"] = ""
                audio_info = torchaudio.info(str(path))
                entry["audio_len"] = audio_info.num_frames / audio_info.sample_rate
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
                else:
                    entry["text"] = ""
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
