from typing import Callable

DATASET_REGISTRY = {}
METHOD_REGISTRY = {}
GENERATOR_REGISTRY = {}


def register_dataset(name: str):
    def deco(fn: Callable):
        DATASET_REGISTRY[name] = fn
        return fn

    return deco


def register_method(name: str):
    def deco(fn: Callable):
        METHOD_REGISTRY[name] = fn
        return fn

    return deco


def register_generator(name: str):
    def deco(fn: Callable):
        GENERATOR_REGISTRY[name] = fn
        return fn

    return deco


def get_dataset_loader(name: str) -> Callable:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def get_method(name: str) -> Callable:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method: {name}. Available: {sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name]


def get_generator(name: str) -> Callable:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator: {name}. Available: {sorted(GENERATOR_REGISTRY)}")
    return GENERATOR_REGISTRY[name]


def register_defaults() -> None:
    # Import side-effect registrations.
    from . import baselines as _baselines  # noqa: F401
    from . import datasets as _datasets  # noqa: F401
    from . import generator as _generator  # noqa: F401
    from . import retrieval as _retrieval  # noqa: F401
