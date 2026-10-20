import re
from collections import defaultdict
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(self, alphabet=None, beam_size=5, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        self.beam_size = beam_size
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        EMPTY_IND = self.char2ind[self.EMPTY_TOK]
        prev = -1
        result = ""
        for ind in inds:
            if ind != EMPTY_IND and ind != prev:
                result += self.ind2char[int(ind)]

            prev = ind
        return result

    def ctc_beam_search(self, probs) -> str:
        """
        Raw decoding without CTC Beam Search.

        Args:
            probs (list): token probabilities
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        EMPTY_IND = self.char2ind[self.EMPTY_TOK]

        beam = {"": [1.0, 0.0]}
        for prob in probs:
            new_beam = defaultdict(lambda: [0.0, 0.0])

            for prefix, (p_b, p_nb) in beam.items():
                for token, p in enumerate(prob):
                    if token == EMPTY_IND:
                        new_beam[prefix][0] += (p_b + p_nb) * p
                        continue

                    char = self.ind2char[token]
                    if prefix and char == prefix[-1]:
                        new_beam[prefix][1] += p_nb * p  # a -> a
                        new_beam[prefix + char][1] += p_b * p  # a -> ^ -> a
                    else:
                        new_beam[prefix + char][1] += (p_b + p_nb) * p

            beam = dict(
                sorted(new_beam.items(), key=lambda x: -sum(x[1]))[: self.beam_size]
            )

        return max(beam, key=lambda prefix: sum(beam[prefix]))

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
