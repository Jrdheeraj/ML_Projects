"""Imbalance handling strategies for fraud training."""

from __future__ import annotations

from dataclasses import dataclass

from imblearn.over_sampling import RandomOverSampler, SMOTE


@dataclass(frozen=True)
class ImbalanceStrategy:
    """Represents sampling strategy with optional sampler object."""

    name: str
    sampler: object | None


def available_strategies(random_state: int) -> list[ImbalanceStrategy]:
    """Return configured imbalance strategies for model comparison."""

    return [
        ImbalanceStrategy(name="none", sampler=None),
        ImbalanceStrategy(
            name="random_oversample",
            sampler=RandomOverSampler(random_state=random_state),
        ),
        ImbalanceStrategy(
            name="smote",
            sampler=SMOTE(random_state=random_state, k_neighbors=3),
        ),
    ]
