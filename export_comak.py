"""Merge the medial and lateral tibial cartilage meshes into one COMAK contact mesh.

`steps/generate_meshes.py` writes each tibial cartilage plate as its own mesh
(`tibia_cart_0_mesh.vtk` = medial, label 5; `tibia_cart_1_mesh.vtk` = lateral,
label 6), because thickness is reported per plate. COMAK uses one tibial
cartilage contact mesh, so this script appends the two into a single file.

The two plates do not touch, so the result is one mesh with two disconnected
pieces. No surface repair is needed.

The pipeline meshes are in image units (mm). OpenSim/COMAK expect metres, so the
default `--scale` is 0.001. Pass `--scale 1` to keep mm.

This is deliberately not a pipeline step: it serves one downstream tool and must
not change what the website receives.

Usage:
    python export_comak.py <working_dir> [--out tibia_cartilage.stl] [--scale 0.001]
"""

import argparse
from pathlib import Path

import vtk

TIBIA_CART_MESHES = ("tibia_cart_0_mesh.vtk", "tibia_cart_1_mesh.vtk")


def _read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError(f"{path} has no points")
    return mesh


def merge_tibia_cartilage(working_dir, out_path=None, scale=0.001) -> Path:
    """Append the two tibial cartilage meshes, scale them, and write one mesh.

    The output format follows the extension of `out_path` (.stl, .vtp or .vtk).
    Returns the path written.
    """
    working_dir = Path(working_dir)
    out_path = Path(out_path) if out_path else working_dir / "tibia_cartilage.stl"

    paths = [working_dir / name for name in TIBIA_CART_MESHES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {working_dir}. Run generate_meshes with "
            "compute_thickness on — cartilage meshes are only saved then."
        )

    append = vtk.vtkAppendPolyData()
    for p in paths:
        append.AddInputData(_read_polydata(p))
    append.Update()

    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    scaler = vtk.vtkTransformPolyDataFilter()
    scaler.SetTransform(transform)
    scaler.SetInputData(append.GetOutput())
    scaler.Update()

    # STL holds triangles only. The pipeline meshes are already triangulated,
    # so this is a safeguard, not a change.
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(scaler.GetOutput())
    tri.Update()

    suffix = out_path.suffix.lower()
    if suffix == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
    elif suffix == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    elif suffix == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    else:
        raise ValueError(f"Unsupported output extension {suffix!r}; use .stl, .vtp or .vtk")
    writer.SetFileName(str(out_path))
    writer.SetInputData(tri.GetOutput())
    writer.Write()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("working_dir", help="Pipeline output directory")
    parser.add_argument("--out", default=None,
                        help="Output file (.stl/.vtp/.vtk). Default: <working_dir>/tibia_cartilage.stl")
    parser.add_argument("--scale", type=float, default=0.001,
                        help="Uniform scale factor. Default 0.001 (mm -> m)")
    args = parser.parse_args()
    out = merge_tibia_cartilage(args.working_dir, args.out, args.scale)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
