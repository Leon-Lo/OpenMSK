"""Tests for export_comak.merge_tibia_cartilage."""

import pytest
import vtk

from export_comak import merge_tibia_cartilage


def _write_sphere(path, center):
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(5.0)
    src.Update()
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(src.GetOutput())
    writer.Write()
    return src.GetOutput()


def _read(path, reader_cls):
    reader = reader_cls()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def test_merges_both_plates_and_scales_to_metres(tmp_path):
    med = _write_sphere(tmp_path / "tibia_cart_0_mesh.vtk", (-20.0, 0.0, 0.0))
    lat = _write_sphere(tmp_path / "tibia_cart_1_mesh.vtk", (20.0, 0.0, 0.0))

    out = merge_tibia_cartilage(tmp_path)

    merged = _read(out, vtk.vtkSTLReader)
    assert out.name == "tibia_cartilage.stl"
    assert merged.GetNumberOfCells() == med.GetNumberOfCells() + lat.GetNumberOfCells()
    # Spheres span x = -25..25 mm, so -0.025..0.025 m after scaling.
    xmin, xmax = merged.GetBounds()[:2]
    assert xmin == pytest.approx(-0.025, abs=1e-6)
    assert xmax == pytest.approx(0.025, abs=1e-6)


def test_scale_one_keeps_mm_and_vtp_output(tmp_path):
    _write_sphere(tmp_path / "tibia_cart_0_mesh.vtk", (-20.0, 0.0, 0.0))
    _write_sphere(tmp_path / "tibia_cart_1_mesh.vtk", (20.0, 0.0, 0.0))

    out = merge_tibia_cartilage(tmp_path, tmp_path / "tc.vtp", scale=1.0)

    merged = _read(out, vtk.vtkXMLPolyDataReader)
    assert merged.GetBounds()[1] == pytest.approx(25.0, abs=1e-4)


def test_missing_plate_raises(tmp_path):
    _write_sphere(tmp_path / "tibia_cart_0_mesh.vtk", (0.0, 0.0, 0.0))
    with pytest.raises(FileNotFoundError, match="tibia_cart_1_mesh.vtk"):
        merge_tibia_cartilage(tmp_path)
