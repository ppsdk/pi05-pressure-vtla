"""Install the upstream openpi Transformers replacements inside the active virtualenv."""

from importlib import metadata
from importlib import util
from pathlib import Path
import shutil
import sys


def main() -> None:
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        raise RuntimeError("Activate a dedicated virtual environment before modifying Transformers")
    version = metadata.version("transformers")
    if version != "4.53.2":
        raise RuntimeError(f"openpi requires transformers==4.53.2; found {version}")
    spec = util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise RuntimeError("Transformers is not installed")
    destination = Path(spec.origin).resolve().parent
    if Path(sys.prefix).resolve() not in destination.parents:
        raise RuntimeError("The Transformers package is outside the active virtual environment")
    source = Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    paths = (
        "models/gemma/configuration_gemma.py",
        "models/gemma/modeling_gemma.py",
        "models/paligemma/modeling_paligemma.py",
        "models/siglip/modeling_siglip.py",
        "models/siglip/check.py",
    )
    for relative in paths:
        src, dst = source / relative, destination / relative
        if not src.is_file() or not dst.parent.is_dir():
            raise FileNotFoundError(f"Missing replacement source or target directory for {relative}")
    for relative in paths:
        shutil.copyfile(source / relative, destination / relative)
    print(f"Installed {len(paths)} openpi replacement files in {destination}")


if __name__ == "__main__":
    main()
