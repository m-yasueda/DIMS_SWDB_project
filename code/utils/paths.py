"""Locating the workshop data, wherever the notebook happens to be running.

The same notebook has to find its data on CodeOcean (`/data/`), off a USB drive handed
out at the workshop (`/Volumes/Brain2026/`), and in a local checkout. Rather than repeat
that ladder in every module, call:

    data_root = resolve_data_root(f"v1dd_{mat_version}_ccm")
    ccm_dir = resolve_dataset_dir(f"v1dd_{mat_version}_ccm", root=data_root)

Precedence, highest first:

1. ``SWDB_DATA_ROOT``, if set. An explicit override always wins.
2. A ``data/<probe>`` directory in the repo, found by searching *upward* from the working
   directory — `data/` sits at the repo root, two levels above a notebook in
   `code/workshops/`. This is the step that makes a local checkout work with no
   configuration at all, which matters because an environment variable only reaches the
   kernel if it was exported in the shell that launched Jupyter, and an IDE-launched
   kernel usually inherits nothing.
3. The platform default: the Brain2026 drive, or ``/data/`` on CodeOcean.

Note the asymmetry with `utils`, which needs none of this: a notebook's kernel starts in
the notebook's own directory (nbconvert sets it from the notebook path; VS Code's
`jupyter.notebookFileRoot` defaults to `${fileDirname}`), and `utils` is a fixed sibling,
so `sys.path.append(pjoin("..", "utils"))` is enough from both `code/workshops/` and
`code/solutions/`. If you deliberately run with a different working directory — say
`notebookFileRoot` set to `${workspaceFolder}` — append `code/utils` instead; the data
lookup below is unaffected either way.
"""

import os
import platform
from os.path import join as pjoin
from typing import Optional

__all__ = ["find_up", "resolve_data_root", "resolve_dataset_dir", "DRIVE_NAME"]

# The USB drive handed out at the workshop. One place to bump it next year.
DRIVE_NAME = "Brain2026"


def find_up(relative_path: str, max_levels: int = 4) -> Optional[str]:
    """Search the working directory and its parents for `relative_path`.

    Pass a distinctive *nested* path (`"code/utils"`, `"data/v1dd_1196_ccm"`) rather than
    a bare directory name, so walking upward cannot match something outside the repo.

    Returns an absolute path, or None if nothing matched.
    """
    for level in range(max_levels + 1):
        candidate = pjoin(os.getcwd(), *([".."] * level), relative_path)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return None


def platform_data_root() -> str:
    """Where the data lives on this kind of machine, ignoring what is actually present."""
    system = platform.system()
    if system == "Darwin":
        # macOS
        return f"/Volumes/{DRIVE_NAME}/"
    if system == "Windows":
        # Windows (replace with the drive letter of the USB drive)
        return "E:/"
    if "amzn" in platform.platform():
        # then on CodeOcean
        return "/data/"
    # then your own linux platform
    # EDIT location where you mounted hard drive
    return f"/media/$USERNAME/{DRIVE_NAME}/"


def resolve_data_root(probe: Optional[str] = None, max_levels: int = 4) -> str:
    """The directory the datasets sit inside. See the module docstring for precedence.

    Parameters
    ----------
    probe:
        Name of a dataset directory to look for under a repo-local ``data/``, e.g.
        ``"v1dd_1196_ccm"``. Omit it to skip the repo-local step.
    max_levels:
        How far up from the working directory to search.

    Returns
    -------
    A path, which is *not* guaranteed to exist — the platform default is a guess about
    where a drive would be mounted. Use `resolve_dataset_dir` to get a checked path.
    """
    if os.environ.get("SWDB_DATA_ROOT"):
        return os.environ["SWDB_DATA_ROOT"]

    if probe:
        local = find_up(pjoin("data", probe), max_levels=max_levels)
        if local:
            return os.path.dirname(local)

    return platform_data_root()


def resolve_dataset_dir(
    *names: str,
    root: Optional[str] = None,
    required: bool = True,
) -> Optional[str]:
    """The first of `names` that exists under `root`, as an absolute path.

    Several names lets a notebook accept more than one layout — on CodeOcean the
    two-photon correlations are attached as their own dataset, while a local download may
    put them beside the EM tables:

        functional_dir = resolve_dataset_dir(
            "v1dd_1196_coreg_functional_correlation", "v1dd_1196", root=data_root
        )

    Raises FileNotFoundError naming every path tried, so a missing dataset fails here
    with something actionable rather than deep inside a later `read_feather`. Pass
    ``required=False`` to get None instead.
    """
    if not names:
        raise ValueError("give at least one dataset name")
    if root is None:
        root = resolve_data_root(names[0])

    tried = []
    for name in names:
        candidate = pjoin(root, name)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
        tried.append(candidate)

    if not required:
        return None
    raise FileNotFoundError(
        "None of these dataset directories exist:\n  "
        + "\n  ".join(tried)
        + f"\n(working directory: {os.getcwd()})\n"
        "Attach the dataset on CodeOcean, mount the "
        f"{DRIVE_NAME} drive, or set SWDB_DATA_ROOT to the directory that contains it."
    )
