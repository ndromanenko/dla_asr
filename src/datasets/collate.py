import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    audios = [item["audio"].squeeze(0) for item in dataset_items]
    spectrogram = [item["spectrogram"].squeeze(0).T for item in dataset_items]
    audio_path = [item["audio_path"] for item in dataset_items]

    spectrogram_length = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items]
    )
    text = [item["text"] for item in dataset_items]
    text_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    text_encoded_length = torch.tensor(
        [item["text_encoded"].shape[-1] for item in dataset_items]
    )

    audio_lengths = torch.tensor([audio.shape[0] for audio in audios])

    pad_audio = pad_sequence(audios, batch_first=True, padding_value=0)
    pad_text_encoded = pad_sequence(text_encoded, batch_first=True, padding_value=0)
    pad_spectrogram = pad_sequence(
        spectrogram, batch_first=True, padding_value=0
    ).transpose(1, 2)

    return {
        "audio": pad_audio,
        "audio_lengths": audio_lengths,
        "spectrogram_length": spectrogram_length,
        "spectrogram": pad_spectrogram,
        "text_encoded": pad_text_encoded,
        "text_encoded_length": text_encoded_length,
        "text": text,
        "audio_path": audio_path,
    }
