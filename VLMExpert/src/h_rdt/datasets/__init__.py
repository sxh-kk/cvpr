"""Project datasets package with Hugging Face datasets symbols re-exported."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import site
import sys


def _load_hf_datasets():
    current_module = sys.modules[__name__]
    for site_dir in site.getsitepackages():
        init_path = Path(site_dir) / "datasets" / "__init__.py"
        if not init_path.exists():
            continue

        spec = spec_from_file_location(
            "datasets",
            init_path,
            submodule_search_locations=[str(init_path.parent)],
        )
        if spec is None or spec.loader is None:
            continue

        module = module_from_spec(spec)
        try:
            sys.modules["datasets"] = module
            spec.loader.exec_module(module)
            return module
        finally:
            sys.modules["datasets"] = current_module
    return None


_HF_DATASETS = _load_hf_datasets()
if _HF_DATASETS is not None:
    for _name in dir(_HF_DATASETS):
        if _name.startswith("_"):
            continue
        globals().setdefault(_name, getattr(_HF_DATASETS, _name))

