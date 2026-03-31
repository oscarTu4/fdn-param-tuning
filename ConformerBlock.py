import torch
import torch.nn as nn
from typing import Optional, Tuple

def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
        converts sequence lengths to a boolean padding mask
        args:
        - lengths: tensor of shape (batch_size,) with sequence lengths
        returns:
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask

class ConvBlock(nn.Module):
    """
        convolutional block used inside the conformer layer
        args:
        - input_dim: input dimension
        - num_channels: number of channels in depthwise convolution
        - kernel_size: kernel size of the depthwise convolution (must be odd)
        - dropout: dropout probability
        - bias: whether to use bias in convolution layers
    """
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel_size must be odd")
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels),
            nn.SiLU(),
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    """
        single conformer layer as detailed in Figure 2b in the paper
        args:
        - input dim: input dimension
        - ffn_dim: hidden layer dimension of the feed forwards
        - num_attention_heads: number multihead attention heads
        - kernel_size: kernel size of the convolutional block
    """
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ffn_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim, bias=True),
            nn.Dropout(dropout),
        )

        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.self_attn = nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_block = ConvBlock(
            input_dim=input_dim,
            num_channels=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            bias=True,
        )

        self.final_layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        
        # half feed forward
        x = x + 0.5 * self.feed_forward(x)

        # multihead attention
        res = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + res

        # convolutional block
        res = x
        x = x.transpose(0, 1)
        x = self.conv_block(x)
        x = x.transpose(0, 1)
        x = x + res

        # half feed forward
        x = x + 0.5 * self.feed_forward(x)

        x = self.final_layer_norm(x)
        return x

class Conformer(nn.Module):
    """
        Main function of the Conformer Block. 
        args:
        - input dim: input dimension
        - num_heads: number of multihead attention heads
        - ffn_dim: hidden layer dimension of the feed forwards
        - num_layers: number of Conformer layers
        - kernel_size: kernel size of the convolutional block
        - dropout: dropout
    """
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # init of conformer layers
        self.conformer_layers = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    kernel_size,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            forward pass of the conformer encoder
            args:
            - input: tensor of shape (batch_size, time, input_dim)
            - lengths: tensor of shape (batch_size,) with sequence lengths
            returns:
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths