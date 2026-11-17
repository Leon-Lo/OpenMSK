"""Run the full pipeline with the inputs set as variables instead of CLI arguments.

Edit the values in the USER SETTINGS block below, then run:
    python run_pipeline_vars.py

This is a thin wrapper around `run_pipeline.run_all` -- it holds no pipeline
logic of its own, so it cannot drift from `run_pipeline.py`.
"""

import os
import shutil
import sys
from pathlib import Path


def _put_this_env_first_on_path():
    """Do what `conda activate` would: put this interpreter's env dirs first on PATH.

    The nnU-Net step shells out to `nnUNetv2_predict`, and Windows resolves that
    name against the *calling* process's PATH. Running this env's python.exe
    without activating the env leaves its Scripts dir off PATH, so another
    install's nnUNetv2_predict.exe (e.g. base Miniconda's) runs instead -- and
    fails with exit 0xC0000022 and no output.
    """
    prefix = Path(sys.prefix)
    dirs = [prefix, prefix / "Scripts", prefix / "Library" / "bin",
            prefix / "Library" / "usr" / "bin", prefix / "Library" / "mingw-w64" / "bin"]
    dirs = [str(d) for d in dirs if d.is_dir()]
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


_put_this_env_first_on_path()

# The step subprocesses (and nnU-Net's inference wrapper) print emoji. On a
# Windows console Python defaults to cp1252: children crash encoding them, and
# this process crashes decoding their captured output. UTF-8 mode is fixed at
# interpreter startup and cannot be switched on afterwards, so if it is off,
# re-run this script once with PYTHONUTF8=1 (children inherit it).
if not sys.flags.utf8_mode:
    import subprocess
    sys.exit(subprocess.call([sys.executable, *sys.argv], env={**os.environ, "PYTHONUTF8": "1"}))

from run_pipeline import run_all  # noqa: E402  (after the PATH fix on purpose)
from steps._common import load_config  # noqa: E402

# ============================== USER SETTINGS ==============================
# These are the committed defaults. Put YOUR values in local_settings.py next
# to this file (git-ignored, so data paths never reach the repo). Any name set
# there overrides the default below, e.g.:
#     PATH_IMAGE = r"D:\scans\knee01\Series 5001"
#     EXPORT_COMAK = True

# Input MRI: a DICOM directory, a NIfTI / NRRD file, or a 3D Slicer .mrb.
# For DICOM, point at ONE series folder (the one holding the .dcm files), not
# the study folder above it. Use the 3D sagittal series (e.g. DESS/qDESS).
PATH_IMAGE = None

# Output directory. Each run goes into its own new subfolder,
# <PATH_SAVE>/<input name>_<YYYYMMDD-HHMMSS>, so a rerun never lands on top of
# an earlier run's files (label_remap refuses to remap a folder twice).
PATH_SAVE = r"results"

# Segmentation model, e.g. "nnunet_knee". None = config's default_seg_model.
MODEL_NAME = None

# Pipeline config. None = KNEEPIPELINE_CONFIG env var, else ./config.json.
CONFIG_PATH = None

# Only used when PATH_IMAGE is a Slicer .mrb bundle that holds more than one
# volume: the file name (or part of it) of the volume to analyse, e.g.
# "qDESS_sag". None is fine when the bundle holds a single volume.
MRB_VOLUME = None

# Also write tibia_cartilage.stl: medial + lateral tibial cartilage merged into
# one contact mesh, in metres, for COMAK. True / False override the config's
# export_comak_tibia_cartilage. None = use the config's value (off if absent).
EXPORT_COMAK = None
# ===========================================================================

try:
    from local_settings import *  # noqa: E402,F401,F403  (personal overrides)
except ImportError:
    pass

# Volume formats Slicer may store inside an .mrb. Segmentations (*.seg.nrrd)
# and label maps are skipped -- the pipeline wants the MRI, not a mask.
_MRB_VOLUME_EXTS = (".nrrd", ".nhdr", ".nii.gz", ".nii", ".mha", ".mhd")
_MRB_SKIP_MARKERS = (".seg.", "label", "segmentation")


def _extract_mrb_volume(mrb_path: Path, save_path: Path) -> Path:
    """Pull the MRI volume out of a Slicer .mrb and write it as .nrrd.

    An .mrb is a zip of the scene. The chosen volume is re-written through
    SimpleITK (rather than copied) so detached-header formats like .nhdr+.raw
    come out as one .nrrd the segment step can read.
    """
    import tempfile
    import zipfile

    import SimpleITK as sitk

    out_path = save_path / f"{mrb_path.stem}.nrrd"
    if out_path.exists():
        return out_path

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(mrb_path) as zf:
            zf.extractall(tmp)

        candidates = [
            p for p in Path(tmp).rglob("*")
            if p.is_file()
            and p.name.lower().endswith(_MRB_VOLUME_EXTS)
            and not any(m in p.name.lower() for m in _MRB_SKIP_MARKERS)
        ]
        if MRB_VOLUME:
            candidates = [p for p in candidates if MRB_VOLUME.lower() in p.name.lower()]

        names = sorted(p.name for p in candidates)
        if len(candidates) != 1:
            raise ValueError(
                f"{mrb_path.name}: expected exactly one volume, found {len(candidates)}: "
                f"{names}. Set MRB_VOLUME to pick one."
            )

        image = sitk.ReadImage(str(candidates[0]))
        sitk.WriteImage(image, str(out_path), useCompression=True)
        print(f"Extracted '{candidates[0].name}' from {mrb_path.name} -> {out_path}")

    return out_path


def _stage_input(input_path: Path, save_path: Path) -> None:
    """Make the input image visible inside the working directory.

    Tries a symlink first (as run_pipeline.py does). On Windows that needs
    Developer Mode or admin rights, so it falls back to copying.
    """
    if input_path.resolve() == save_path.resolve() or input_path.resolve().is_relative_to(save_path.resolve()):
        return
    target = save_path / input_path.name
    if target.exists():
        return
    try:
        target.symlink_to(input_path.resolve(), target_is_directory=input_path.is_dir())
    except OSError:
        if input_path.is_dir():
            shutil.copytree(input_path, target)
        else:
            shutil.copy2(input_path, target)


def main():
    config_path = CONFIG_PATH or os.environ.get(
        "KNEEPIPELINE_CONFIG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
    )
    config = load_config(config_path)
    if EXPORT_COMAK is not None:
        config["export_comak_tibia_cartilage"] = EXPORT_COMAK

    from datetime import datetime

    if PATH_IMAGE is None:
        raise ValueError("Set PATH_IMAGE in local_settings.py (next to this script).")
    input_path = Path(PATH_IMAGE)
    if not input_path.exists():
        raise FileNotFoundError(f"PATH_IMAGE does not exist: {input_path}")
    run_name = input_path.name.split(".")[0] if input_path.is_file() else input_path.name
    save_path = Path(PATH_SAVE) / f"{run_name}_{datetime.now():%Y%m%d-%H%M%S}"
    save_path.mkdir(parents=True, exist_ok=False)
    print(f"Writing results to: {save_path.resolve()}")

    if input_path.is_dir() and not any(f.is_file() for f in input_path.iterdir()):
        series = sorted(p.name for p in input_path.iterdir() if p.is_dir())
        raise ValueError(
            f"PATH_IMAGE has no files of its own, only subfolders: {series}. "
            f"Point it at the one series folder that holds the .dcm files."
        )

    if input_path.suffix.lower() == ".mrb":
        _extract_mrb_volume(input_path, save_path)
    else:
        _stage_input(input_path, save_path)
    run_all(save_path, model_name=MODEL_NAME, config=config, config_path=config_path)


if __name__ == "__main__":
    main()
