#!/usr/bin/env python
"""Convert a mesh into a MuJoCo/mjlab-friendly USD asset.

This converter is intentionally Isaac-free. It loads the source mesh with
``trimesh`` and writes a compact USD stage with a single mesh prim plus an
optional preview material/texture network. The output is suitable for
``mj_loadUSD``-based consumers such as mjlab.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def _safe_token(name: str, fallback: str = "asset") -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not token:
        token = fallback
    if token[0].isdigit():
        token = f"_{token}"
    return token


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(loaded)!r}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no geometry: {mesh_path}")
    return loaded


def _rgba_from_any(color) -> tuple[float, float, float, float]:
    if color is None:
        return (0.78, 0.78, 0.78, 1.0)
    arr = np.asarray(color, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return (0.78, 0.78, 0.78, 1.0)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1.0:
        arr = arr / 255.0
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size == 2:
        arr = np.array([arr[0], arr[0], arr[0], arr[1]], dtype=np.float64)
    elif arr.size == 3:
        arr = np.concatenate([arr, [1.0]])
    return tuple(float(x) for x in arr[:4])


def _vec3f_array(points: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])


def _vec2f_array(points: np.ndarray) -> Vt.Vec2fArray:
    return Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in points])


def _save_texture(image, dest_path: Path, source_path: Path | None = None) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path is not None and source_path.exists():
        shutil.copy2(source_path, dest_path)
        return

    if hasattr(image, "save"):
        pil_image = image
        mode = getattr(pil_image, "mode", None)
        if mode not in {"RGB", "RGBA"}:
            try:
                bands = set(pil_image.getbands())
                pil_image = pil_image.convert("RGBA" if "A" in bands else "RGB")
            except Exception:
                pass
        pil_image.save(dest_path)
        return

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Pillow is required to write texture images") from exc

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        Image.fromarray(array).save(dest_path)
    elif array.shape[-1] == 4:
        Image.fromarray(array, mode="RGBA").save(dest_path)
    else:
        Image.fromarray(array, mode="RGB").save(dest_path)


def _build_preview_material(
    stage: Usd.Stage,
    material_path: str,
    diffuse_rgb: tuple[float, float, float],
    texture_rel_path: str | None,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    preview = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    if texture_rel_path:
        st_reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReaderSt")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        tex_shader = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
        tex_shader.CreateIdAttr("UsdUVTexture")
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_rel_path))
        tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        rgb_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        alpha_output = tex_shader.CreateOutput("a", Sdf.ValueTypeNames.Float)
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_output)
        preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(alpha_output)
    else:
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_rgb))

    surface_output = preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    return material


def convert_mesh(input_path: Path, output_path: Path, scale: float) -> None:
    mesh = _load_mesh(input_path)

    vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(scale)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    asset_name = _safe_token(output_path.stem, "asset")
    root = UsdGeom.Xform.Define(stage, f"/{asset_name}")
    stage.SetDefaultPrim(root.GetPrim())

    mesh_prim = UsdGeom.Mesh.Define(stage, f"/{asset_name}/mesh")
    mesh_prim.CreatePointsAttr(_vec3f_array(vertices))
    mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh_prim.CreateDoubleSidedAttr().Set(True)
    mesh_prim.CreateExtentAttr().Set(
        _vec3f_array(
            np.asarray(
                [
                    vertices.min(axis=0),
                    vertices.max(axis=0),
                ],
                dtype=np.float32,
            )
        )
    )

    if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(vertices):
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        mesh_prim.CreateNormalsAttr(_vec3f_array(normals))
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    diffuse = None
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is not None:
        diffuse = getattr(material, "diffuse", None)
    if diffuse is None and getattr(mesh.visual, "main_color", None) is not None:
        diffuse = mesh.visual.main_color
    diffuse_rgb = _rgba_from_any(diffuse)[:3]

    texture_rel_path = None
    uv = getattr(mesh.visual, "uv", None)
    image = getattr(material, "image", None) if material is not None else None
    image_filename = getattr(image, "filename", None) if image is not None else None

    if uv is not None and image is not None:
        uv = np.asarray(uv, dtype=np.float32)
        if len(uv) == len(vertices):
            face_uv = uv[faces.reshape(-1)]
        elif len(uv) == len(faces) * 3:
            face_uv = uv
        else:
            face_uv = None

        if face_uv is not None:
            primvars = UsdGeom.PrimvarsAPI(mesh_prim)
            st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            st.Set(_vec2f_array(face_uv))

            texture_dir = output_path.parent / "textures" / asset_name
            if image_filename:
                source_path = Path(image_filename)
                if not source_path.is_absolute():
                    source_path = (input_path.parent / source_path).resolve()
                if source_path.exists():
                    texture_suffix = source_path.suffix.lower() or ".png"
                    texture_name = _safe_token(source_path.stem, asset_name)
                    texture_rel_path = f"textures/{asset_name}/{texture_name}{texture_suffix}"
                    _save_texture(image, texture_dir / f"{texture_name}{texture_suffix}", source_path)
                else:
                    texture_rel_path = f"textures/{asset_name}/{asset_name}_baseColor.png"
                    _save_texture(image, texture_dir / f"{asset_name}_baseColor.png")
            else:
                texture_rel_path = f"textures/{asset_name}/{asset_name}_baseColor.png"
                _save_texture(image, texture_dir / f"{asset_name}_baseColor.png")

    material_prim = f"/{asset_name}/Looks/Material"
    preview_material = _build_preview_material(stage, material_prim, diffuse_rgb, texture_rel_path)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(preview_material)

    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a mesh file into a USD asset.")
    parser.add_argument("input", type=Path, help="Path to the input mesh file.")
    parser.add_argument("output", type=Path, help="Path to the output USD file.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor for the mesh.")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--make-instanceable", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--collision-approximation", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mass", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    convert_mesh(args.input.resolve(), args.output.resolve(), args.scale)


if __name__ == "__main__":
    main()
