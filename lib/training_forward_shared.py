"""Exact int64 projection with one shared histogram per warp.

The fixed measured configuration uses 16^3 source cubes, 64 threads and two
256-bin integer caches. Each contribution is quantized before integer addition.
Cache misses still use global atomics; signed contributions remain supported.
NVRTC and CUDA-driver dependencies are unchanged from the production backend.
"""
import ctypes as C
from pathlib import Path

import torch
if __package__:
    from .nvrtc_driver import CudaModule, compile_ptx
else:
    from nvrtc_driver import CudaModule, compile_ptx

_modules = {}
_compilation_info = {}
_compiled_ptx = {}


def project_into(pc, sens_x, sens_y, sens_z, histogram, *, grid_size=400,
                 r_min, delta_r, voxel_size, center, fixed_scale):
    """Accumulate into the caller's contiguous, zeroed, full int64 histogram."""
    if grid_size <= 0 or grid_size % 16 or pc.numel() != grid_size ** 3:
        raise ValueError("a complete positive grid divisible by 16 is required")
    tensors = (pc, sens_x, sens_y, sens_z, histogram)
    if any(not t.is_cuda or not t.is_contiguous() or t.device != pc.device for t in tensors):
        raise ValueError("all tensors must be contiguous and on one CUDA device")
    if any(t.dtype != torch.float32 for t in tensors[:-1]) or histogram.dtype != torch.int64:
        raise ValueError("float32 inputs and int64 histogram required")
    if histogram.ndim != 2 or min(histogram.shape) <= 0:
        raise ValueError("histogram must have positive shape (n_sensors, n_bins)")
    n_sensors, n_bins = histogram.shape
    if any(t.numel() != n_sensors for t in (sens_x, sens_y, sens_z)):
        raise ValueError("sensor count mismatch")
    key = (pc.device.index, grid_size, n_sensors)
    if key not in _modules:
        source_path = Path(__file__).with_suffix(".cu")
        options = ["-DCUBE_SIZE=16", "-DCACHE_BINS=256", "-DCACHE_OFFSET=109",
                   f"-DFIXED_GRID_SIZE={grid_size}", f"-DFIXED_N_SENSORS={n_sensors}"]
        ptx, info = compile_ptx(source_path.read_text(), options=options, name=source_path.name)
        _modules[key] = CudaModule(ptx, device=pc.device)
        _compilation_info[str(key)] = info
        _compiled_ptx[str(key)] = ptx
    arguments = ([C.c_void_p(t.data_ptr()) for t in tensors]
                 + [C.c_int(grid_size), C.c_int(n_sensors), C.c_int(n_bins)]
                 + [C.c_float(x) for x in (r_min, delta_r, voxel_size, center, fixed_scale)])
    _modules[key].launch("project_cube_shared", ((grid_size // 16) ** 3 * n_sensors,),
                         (64,), arguments)
    return histogram


def project_looped_into(pc, sens_x, sens_y, sens_z, histogram, *, grid_size=400,
                        voxel_size, center, r_min, delta_r, fixed_scale,
                        tiles_per_cta=4, relaxed=False):
    """Compatible production entry point; legacy scheduling flags are ignored.

    Dispatch/architecture fallback remains the caller's responsibility. Uses
    relaxed device atomics and PyTorch's current stream, as in the old backend.
    """
    return project_into(pc, sens_x, sens_y, sens_z, histogram, grid_size=grid_size,
                        voxel_size=voxel_size, center=center, r_min=r_min, delta_r=delta_r,
                        fixed_scale=fixed_scale)
