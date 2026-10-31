"""Install wheel and sdist separately and run examples outside the checkout.

Requires a working pip and access to the configured package index. No upload.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import tarfile
import venv
import zipfile


PACKAGE = "modaldyn"
MODULE = "modaldyn"
EXAMPLES = ("quickstart.py", "controlled.py")


def run(python, *args, cwd):
    subprocess.run([str(python), *args], cwd=cwd, check=True)


def source_version(root):
    tree = ast.parse((root / "src" / MODULE / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("Missing package version")


def verify_contents(artifact, root):
    """Compare shipped code and required source files to this checkout."""
    package_files = [p for p in (root / "src" / MODULE).rglob("*")
                     if p.is_file() and (p.suffix == ".py" or p.name == "py.typed")]
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            for path in package_files:
                name = path.relative_to(root / "src").as_posix()
                if archive.read(name) != path.read_bytes():
                    raise RuntimeError(f"Wheel source mismatch: {name}")
            if not any(name.endswith("/LICENSE") for name in archive.namelist()):
                raise RuntimeError("Wheel is missing its license")
        return len(package_files)
    required = package_files + [root / name for name in ("README.md", "LICENSE", "pyproject.toml", "CHANGELOG.md")]
    for pattern in ("docs/**/*.md", "tests/*.py", "examples/*.py", "scripts/*.py", ".github/workflows/*.yml"):
        required.extend(root.glob(pattern))
    with tarfile.open(artifact) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        for path in required:
            name = path.relative_to(root).as_posix()
            member = archive.extractfile(prefix + "/" + name)
            if member is None or member.read() != path.read_bytes():
                raise RuntimeError(f"Source archive mismatch: {name}")
    return len(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = source_version(root)
    directory = args.dist_dir.resolve()
    artifacts = sorted(directory.glob("*.whl")) + sorted(directory.glob("*.tar.gz"))
    stem = PACKAGE.replace("-", "_") + "-" + version
    expected = {stem + "-py3-none-any.whl", stem + ".tar.gz"}
    if {path.name for path in artifacts} != expected:
        raise RuntimeError(f"Expected exactly the candidate wheel and sdist: {sorted(expected)}")
    records = []
    for artifact in artifacts:
        verified_files = verify_contents(artifact, root)
        with tempfile.TemporaryDirectory(prefix=MODULE + "-artifact-") as temporary:
            outside = Path(temporary)
            environment = outside / "venv"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / ("Scripts/python.exe" if (environment / "Scripts").exists()
                                    else "bin/python")
            run(python, "-m", "pip", "install", "--disable-pip-version-check", str(artifact), cwd=outside)
            run(python, "-m", "pip", "check", cwd=outside)
            smoke = (
                "import importlib, importlib.metadata, importlib.util, pathlib, sys; "
                f"m=importlib.import_module({MODULE!r}); "
                f"assert m.__version__ == importlib.metadata.version({PACKAGE!r}) == {version!r}; "
                "assert pathlib.Path(m.__file__).is_relative_to(pathlib.Path(sys.prefix)); "
                "assert importlib.util.find_spec('torch') is None; "
                "print('Clean base installation:', m.__version__)"
            )
            run(python, "-I", "-c", smoke, cwd=outside)
            for example in EXAMPLES:
                run(python, "-I", str(root / "examples" / example), cwd=outside)
            records.append({
                "artifact": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "version": version, "examples": list(EXAMPLES), "status": "passed",
                "torch_installed": False,
                "source_files_verified": verified_files,
            })
    report = {"package": PACKAGE, "artifacts": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
