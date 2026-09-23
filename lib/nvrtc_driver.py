"""Compile CUDA using the pinned NVRTC wheel; launch on PyTorch's current stream.

This experiment needs no nvcc, C++ compiler, PyTorch extension, or new packages.
Compilation is CPU-only. A module uses PyTorch's existing primary CUDA context;
keep it alive for as long as a captured graph can reference its functions.
"""
import ctypes as C
import importlib.util
from pathlib import Path


def _bind(library, name, arguments, result=C.c_int):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


def compile_ptx(source, *, architecture="compute_80", options=(), name="projection.cu"):
    spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("The pinned nvidia-cuda-nvrtc-cu12 package is unavailable")
    paths = [Path(root) / "lib/libnvrtc.so.12" for root in spec.submodule_search_locations]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise RuntimeError("Cannot locate libnvrtc.so.12 in the pinned NVRTC package")
    library = C.CDLL(str(path))
    handle = C.c_void_p()
    charpp = C.POINTER(C.c_char_p)
    create = _bind(library, "nvrtcCreateProgram", [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                                  C.c_int, charpp, charpp])
    compile_program = _bind(library, "nvrtcCompileProgram", [C.c_void_p, C.c_int, charpp])
    log_size = _bind(library, "nvrtcGetProgramLogSize", [C.c_void_p, C.POINTER(C.c_size_t)])
    get_log = _bind(library, "nvrtcGetProgramLog", [C.c_void_p, C.c_void_p])
    ptx_size = _bind(library, "nvrtcGetPTXSize", [C.c_void_p, C.POINTER(C.c_size_t)])
    get_ptx = _bind(library, "nvrtcGetPTX", [C.c_void_p, C.c_void_p])
    destroy = _bind(library, "nvrtcDestroyProgram", [C.POINTER(C.c_void_p)])
    error_string = _bind(library, "nvrtcGetErrorString", [C.c_int], C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(error_string(status).decode())

    check(create(C.byref(handle), source.encode(), name.encode(), 0, None, None))
    arguments = [f"--gpu-architecture={architecture}", "--std=c++17", "--fmad=false", *options]
    encoded = (C.c_char_p * len(arguments))(*(argument.encode() for argument in arguments))
    try:
        status = compile_program(handle, len(arguments), encoded)
        size = C.c_size_t()
        check(log_size(handle, C.byref(size)))
        log = C.create_string_buffer(max(size.value, 1))
        check(get_log(handle, log))
        if status:
            raise RuntimeError(f"NVRTC compilation failed: {error_string(status).decode()}\n{log.value.decode()}")
        check(ptx_size(handle, C.byref(size)))
        output = C.create_string_buffer(size.value)
        check(get_ptx(handle, output))
        return output.raw, {"nvrtc_library": str(path), "options": arguments, "log": log.value.decode()}
    finally:
        destroy(C.byref(handle))


class CudaModule:
    """Module handles are intentionally retained until process exit; no finalizer."""
    def __init__(self, ptx, *, device=None):
        import torch
        self.device = torch.cuda.current_device() if device is None else torch.device(device).index
        if self.device is None:
            self.device = torch.cuda.current_device()
        self.driver = C.CDLL("libcuda.so.1")
        self._error = _bind(self.driver, "cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)])
        self._load = _bind(self.driver, "cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p])
        self._function = _bind(self.driver, "cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
        self._launch = _bind(self.driver, "cuLaunchKernel", [C.c_void_p, *([C.c_uint] * 7),
                                                             C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p])
        self.module = C.c_void_p()
        self.functions = {}
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device)  # Create PyTorch's primary context first.
            image = C.create_string_buffer(ptx)
            self._check(self._load(C.byref(self.module), image))

    def _check(self, status):
        if status:
            message = C.c_char_p()
            self._error(status, C.byref(message))
            raise RuntimeError(f"CUDA driver error {status}: {message.value.decode() if message.value else 'unknown'}")

    def function(self, name):
        if name not in self.functions:
            handle = C.c_void_p()
            self._check(self._function(C.byref(handle), self.module, name.encode()))
            self.functions[name] = handle
        return self.functions[name]

    def launch(self, name, grid, block, arguments, *, shared_bytes=0):
        """arguments contains ctypes scalars, including c_void_p(tensor.data_ptr())."""
        import torch
        grid = (*grid, 1, 1)[:3]
        block = (*block, 1, 1)[:3]
        if any(int(value) <= 0 for value in (*grid, *block)):
            raise ValueError("Launch dimensions must be positive")
        pointers = (C.c_void_p * len(arguments))(*(C.addressof(value) for value in arguments))
        with torch.cuda.device(self.device):
            stream = C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            self._check(self._launch(self.function(name), *grid, *block, shared_bytes,
                                     stream, pointers, None))
