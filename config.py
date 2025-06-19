from dataclasses import dataclass
import typing as tp


@dataclass
class Config:
    seed: int = 24
    batch_size: int = 8
    n_epochs: int = 15
    lr: float = 1e-4
    scheduler_type: tp.Literal['exp', 'step', 'cosine'] | None = "cosine"
