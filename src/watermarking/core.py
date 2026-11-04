import itertools
from collections import OrderedDict
from dataclasses import dataclass
from math import erfc, floor, isfinite, sqrt

import torch
from transformers import LogitsProcessor


@dataclass(frozen=True)
class WatermarkConfig:
    vocab_size: int
    gamma: float = 0.25
    delta: float = 2.0
    key: int = 15485863

    def __post_init__(self):
        if self.vocab_size < 2 or not 0 < self.gamma < 1:
            raise ValueError("Require vocab_size >= 2 and 0 < gamma < 1")
        if not 0 < floor(self.vocab_size * self.gamma) < self.vocab_size:
            raise ValueError("Green and red lists must both be nonempty")
        if not isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and nonnegative")
        if not 0 < self.key < 2**63:
            raise ValueError("key must be a positive signed 64-bit integer")


class Greenlist:
    """simple_1 using the CPU PyTorch PRNG, independent of inference device."""

    def __init__(self, config: WatermarkConfig, cache_size: int = 128):
        self.config = config
        self.generator = torch.Generator(device="cpu")
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def ids(self, previous: int) -> torch.Tensor:
        if not 0 <= previous < self.config.vocab_size:
            raise ValueError("Token ID outside the watermark vocabulary")
        if previous in self.cache:
            self.cache.move_to_end(previous)
            return self.cache[previous]
        self.generator.manual_seed((self.config.key * previous) % (2**64))
        ids = torch.randperm(self.config.vocab_size, generator=self.generator)[
            : floor(self.config.vocab_size * self.config.gamma)
        ]
        if self.cache_size:
            self.cache[previous] = ids
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return ids


class SoftWatermark(LogitsProcessor):
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.greenlist = Greenlist(config)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Expected nonempty [batch, sequence] IDs and [batch, vocab] scores")
        if input_ids.shape[0] != scores.shape[0] or scores.shape[1] != self.config.vocab_size:
            raise ValueError("Batch size or vocabulary does not match watermark configuration")
        result = scores.clone()
        if self.config.delta:
            for batch, previous in enumerate(input_ids[:, -1].tolist()):
                ids = self.greenlist.ids(previous).to(scores.device)
                result[batch, ids] += self.config.delta
        return result


class Detector:
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.greenlist = Greenlist(config)

    def score(self, token_ids, deduplicate: bool = True) -> dict:
        ids = [int(x) for x in token_ids]
        if any(x < 0 or x >= self.config.vocab_size for x in ids):
            raise ValueError("Token ID outside the watermark vocabulary")
        pairs = list(itertools.pairwise(ids))
        if deduplicate:
            pairs = list(dict.fromkeys(pairs))
        count = len(pairs)
        green = sum(int(bool((self.greenlist.ids(a) == b).any())) for a, b in pairs)
        # Retain the authors' nominal gamma convention for reference compatibility.
        z = (
            (green - self.config.gamma * count)
            / sqrt(count * self.config.gamma * (1 - self.config.gamma))
            if count
            else None
        )
        return {
            "status": "ok" if count else "insufficient_text",
            "num_tokens_observed": len(ids),
            "num_tokens_scored": count,
            "num_green_tokens": green,
            "green_fraction": green / count if count else None,
            "z_score": z,
            "p_value": 0.5 * erfc(z / sqrt(2)) if z is not None else None,
            "deduplicate": deduplicate,
            "effective_gamma": floor(self.config.vocab_size * self.config.gamma)
            / self.config.vocab_size,
            "detector_config": {
                "vocab_size": self.config.vocab_size,
                "gamma": self.config.gamma,
                "key": self.config.key,
                "seeding_scheme": "simple_1",
                "partition_device": "cpu",
            },
        }
