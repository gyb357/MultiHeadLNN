import torch.nn as nn
from typing import Dict
from ncps.torch import LTC, CfC


def get_cell(
        type: str,
        input_size: int,
        window_size: int,
        batch_first: bool = True
) -> nn.Module:
    type: str = type.lower()
    types: Dict[str, nn.Module] = {'rnn': nn.RNN,
                                   'lstm': nn.LSTM,
                                   'gru': nn.GRU,
                                   'ltc': LTC,
                                   'cfc': CfC}
    
    # Type check
    if type not in types:
        raise ValueError(f"Unsupported cell type: {type}. Supported types are: {list(types.keys())}")

    # Standard RNN cells
    if type in ['rnn', 'lstm', 'gru']:
        cell = types[type](input_size=input_size,
                           hidden_size=window_size,
                           batch_first=batch_first)
    # NCPS cells
    elif type in ['ltc', 'cfc']:
        cell = types[type](input_size=input_size,
                           units=window_size,
                           batch_first=batch_first)
    return cell


def init_weights(module: nn.Module) -> None:
    # Standard RNN weight initialization
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def get_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

