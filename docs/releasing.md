# Build and publish a release

GitHub source publication and PyPI package publication are separate operations.
Do not announce a PyPI installation command as verified until the project page
and the wheel for that version can be read back from PyPI.

## Validate the exact source

From a clean checkout on the intended release commit:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/quickstart.py
python examples/controlled.py
python -m build
python -m twine check dist/*
python scripts/check_artifacts.py --output artifact-validation.json
```

For PyTorch validation install the intended CPU/CUDA build separately and rerun
the suite. CUDA test skips must remain visible and must not be described as GPU
verification. The manual CUDA CI job requires a self-hosted runner labeled
`self-hosted`, `linux`, `x64`, `gpu` with Python, compatible PyTorch and an NVIDIA
driver already provisioned. Enabling that job asserts CUDA availability before
running tests. It does not provision or pay for a machine.

Verify the built wheel outside the source checkout:

```bash
python -m venv /tmp/modaldyn-wheel-check
/tmp/modaldyn-wheel-check/bin/python -m pip install dist/modaldyn-0.1.2-py3-none-any.whl
cd /tmp
/tmp/modaldyn-wheel-check/bin/python -c 'import modaldyn; print(modaldyn.__version__)'
```

Run an example using that clean interpreter as well. Compare the wheel contents
with the source manifest. Preserve test and benchmark outputs with the release;
include software versions and which GPU checks were skipped.

`scripts/check_artifacts.py` automates independent wheel and source installs in
temporary environments outside the checkout. It checks exact metadata/module
versions, base-only imports, `pip check`, both examples, and distribution hashes.
It requires pip/index access and never uploads. Use a fresh distribution folder;
stale archives from a different version cause validation to fail.

Release 0.1.2 can also be initiated by pushing its validated source to
`release/0.1.2`. This branch trigger accepts only package version 0.1.2,
uses the exact pushed commit, and preserves the existing `pypi`
Trusted Publisher identity. Future versions use the manual tag workflow.
A successful build alone is not proof of an upload; verify PyPI readback.

## GitHub release

The repository is `giovannyTorres/dmd`. Commit validated source, create an
annotated `v0.1.2` tag for that commit, push it, and create a release containing
the wheel, source distribution and checksums. Release notes should identify
the tested CPU/GPU configurations and current limitations. Never replace an
already published version with different code; increment the version instead.

## Configure PyPI Trusted Publishing

A project owner must configure a PyPI publisher at
<https://pypi.org/manage/account/publishing/> (a pending publisher can create a
new project):

| Field | Value |
| --- | --- |
| PyPI project name | `modaldyn` |
| Existing public PyPI account | `GiovannyTorres674` |
| GitHub owner | `giovannyTorres` |
| Repository | `dmd` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Create the GitHub environment named `pypi`. Apply branch/tag protections and
review rules appropriate to the owner. The workflow requests a short-lived
OIDC credential; no long-lived PyPI token belongs in the source or artifacts.
The project already exists on PyPI. Verify its publisher configuration and
version history before starting another publication.

Run **Publish package** manually from GitHub Actions with the exact tag
`v0.1.2`. The workflow checks the tag against package metadata, executes tests,
builds and validates distributions, and only then starts the `pypi` environment
job. The PyPI action uses Trusted Publishing and provenance attestations.

If publishing fails because the account, project name or Trusted Publisher is
not configured, the validated GitHub release remains usable. Report the actual
failure and complete that account setup before retrying; do not infer success
from an uploaded GitHub artifact. Never paste credentials into chat.

After a successful upload, verify the project/version endpoint and install
from the registry in a fresh environment:

```bash
python -m venv /tmp/modaldyn-pypi-check
/tmp/modaldyn-pypi-check/bin/python -m pip install --index-url https://pypi.org/simple modaldyn==0.1.2
/tmp/modaldyn-pypi-check/bin/python -c 'import modaldyn; print(modaldyn.__version__)'
```

Only after this readback should release notes state that `pip install modaldyn`
is available from PyPI.
