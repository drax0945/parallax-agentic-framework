"""Molecular machine learning: models trained on cited data, inside the session.

Parallax does not ship pre-trained black boxes. A model is built when it is
needed, from data pulled live out of ChEMBL, and it is stored with a model card
that lists the source documents behind every training point. A prediction can
therefore be traced: value -> model -> training set -> DOIs.
"""

from .qsar import (
    ENDPOINT_PRESETS,
    ModelCard,
    list_models,
    load_model,
    predict,
    train,
)

__all__ = [
    "ENDPOINT_PRESETS",
    "ModelCard",
    "list_models",
    "load_model",
    "predict",
    "train",
]
