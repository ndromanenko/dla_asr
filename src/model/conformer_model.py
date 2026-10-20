import math

import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        positions = torch.arange(10000).unsqueeze(1)

        div_term = torch.exp(
            -math.log(10000) * torch.arange(0, embed_dim, 2) / embed_dim
        )

        pe = torch.zeros(10000, embed_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear_1 = nn.Linear(embed_dim, hidden_size)
        self.non_linear = nn.SiLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_size, embed_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.non_linear(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, dropout):

        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_1 = FeedForward(embed_dim, embed_dim * 4, dropout)
        self.ffn_2 = FeedForward(embed_dim, embed_dim * 4, dropout)
        self.conv_module = ConvolutionModule(embed_dim, kernel_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        x_old = x + 0.5 * self.ffn_1(x)
        x = self.layer_norm_1(x_old)
        x = self.multihead_attn(x, x, x, padding_mask)[0]
        x = x_old + self.dropout(x)
        x = x + self.conv_module(x, padding_mask)
        x = x + 0.5 * self.ffn_2(x)
        x = self.layer_norm_2(x)

        return x


class ConvolutionModule(nn.Module):

    def __init__(self, embed_dim, kernel_size, dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_1 = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=1)
        self.pointwise_2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            groups=embed_dim,
            padding="same",
        )
        self.glu = nn.GLU(dim=1)
        self.conv_norm = nn.LayerNorm(embed_dim)
        self.non_linear = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):

        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_1(x)
        x = self.glu(x)
        x = x.masked_fill(padding_mask[:, None, :], 0)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.conv_norm(x)
        x = x.transpose(1, 2)
        x = self.non_linear(x)
        x = self.pointwise_2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        return x


class ConvSubsampling(nn.Module):

    def __init__(self, out_chanel, kernel_size, stride, embed_dim, n_mels, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(1, out_chanel, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(
            out_chanel, out_chanel, kernel_size=kernel_size, stride=stride
        )
        self.dropout = nn.Dropout(dropout)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

        self.flatten = nn.Flatten(start_dim=2)

        size_1 = (n_mels - kernel_size) // stride + 1
        size_2 = (size_1 - kernel_size) // stride + 1

        self.linear = nn.Linear(size_2 * out_chanel, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu_1(x)
        x = self.conv2(x)
        x = self.relu_2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        return x


class ConformerModel(nn.Module):
    """
    Conformer model
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        embed_dim,
        num_blocks,
        num_heads,
        conv_kernel_size,
        subsampling_chanels,
        dropout,
    ):

        super().__init__()

        self.conv_subsampling = ConvSubsampling(
            out_chanel=subsampling_chanels,
            kernel_size=3,
            stride=2,
            embed_dim=embed_dim,
            n_mels=n_feats,
            dropout=dropout,
        )
        self.pos_emb = PositionEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Linear(embed_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = self.conv_subsampling(spectrogram)
        x = self.pos_emb(x)

        log_probs_length = self.transform_input_lengths(spectrogram_length)
        padding_mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            >= log_probs_length.to(x.device)[:, None]
        )

        for block in self.blocks:
            x = block(x, padding_mask)

        output = self.head(x)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        lengths = input_lengths
        for _ in range(2):
            lengths = (lengths - 3) // 2 + 1
        return lengths

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
