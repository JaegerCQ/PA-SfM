// Exact int64 histogram aggregation: one 256-bin cache per warp.
// Fixed measured configuration: 16^3 sources, 64 threads, two integer shards.

__device__ __forceinline__ float div_full(float a, float b) {
    float out;
    asm("div.full.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float sqrt_approx(float a) {
    float out;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

__device__ __forceinline__ float floor_ftz(float a) {
    float out;
    asm("cvt.rmi.ftz.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

__device__ __forceinline__ float radius(float ix, float iy, float iz,
                                       float sx, float sy, float sz,
                                       float center, float voxel) {
    float dx = __fmaf_rn(-__fsub_rn(ix, center), voxel, sx);
    float dy = __fmaf_rn(-__fsub_rn(iy, center), voxel, sy);
    float dz = __fmaf_rn(-__fsub_rn(iz, center), voxel, sz);
    float xy = __fmaf_rn(dx, dx, __fmul_rn(dy, dy));
    float xyz = __fmaf_rn(dz, dz, xy);
    return sqrt_approx(__fadd_rn(xyz, 1.0e-12f));
}

__device__ __forceinline__ long long quantize(float unscaled, float scale) {
    // Match the actual original Triton PTX, not an algebraic rewrite:
    // positive branch fuses scale multiplication with +0.5; negative branch
    // rounds the multiplication before adding +0.5.
    float scaled = __fmul_rn(unscaled, scale);
    float positive = floor_ftz(__fmaf_rn(unscaled, scale, 0.5f));
    float negative_scaled = __fmaf_rn(-unscaled, scale, 0.0f);
    float negative = __fsub_rn(0.0f, floor_ftz(__fadd_rn(negative_scaled, 0.5f)));
    return __float2ll_rz(scaled >= 0.0f ? positive : negative);
}

extern "C" __global__ void project_cube_shared(
    const float* __restrict__ pc,
    const float* __restrict__ sx_ptr,
    const float* __restrict__ sy_ptr,
    const float* __restrict__ sz_ptr,
    unsigned long long* __restrict__ histogram,
    int runtime_grid_size, int runtime_n_sensors, int n_bins,
    float r_min, float delta_r, float voxel, float center, float scale
) {
    constexpr int MODE = 2;
#ifdef FIXED_GRID_SIZE
    constexpr int grid_size = FIXED_GRID_SIZE;
    constexpr int n_sensors = FIXED_N_SENSORS;
#else
    int grid_size = runtime_grid_size;
    int n_sensors = runtime_n_sensors;
#endif
    constexpr int THREADS = 64;
    constexpr int LOCAL_BINS = CACHE_BINS;
    constexpr int HIST_SHARDS = 2;
    __shared__ unsigned long long local_hist[LOCAL_BINS * HIST_SHARDS];
    __shared__ int local_base;
    int sensor = blockIdx.x % n_sensors;
    int group = blockIdx.x / n_sensors;
    int lane = threadIdx.x;
    float sx = sx_ptr[sensor], sy = sy_ptr[sensor], sz = sz_ptr[sensor];
    int cube_x = 0, cube_y = 0, cube_z = 0;
    if constexpr (MODE != 0) {
        int cubes = grid_size / CUBE_SIZE;
        cube_x = (group / (cubes * cubes)) * CUBE_SIZE;
        cube_y = ((group / cubes) % cubes) * CUBE_SIZE;
        cube_z = (group % cubes) * CUBE_SIZE;
    }
    if constexpr (MODE == 2) {
        for (int bin = lane; bin < LOCAL_BINS * HIST_SHARDS; bin += THREADS) local_hist[bin] = 0ULL;
        if (lane == 0) {
            // Cube bounds are specialized to the production
            // voxel/delta ratio. Five extra bins cover FP rounding.
            // Any deposit outside this cache falls back to a global atomic,
            // so even unusual grid/voxel parameters cannot discard a source.
            float r_center = radius(cube_x + 0.5f * (CUBE_SIZE - 1), cube_y + 0.5f * (CUBE_SIZE - 1), cube_z + 0.5f * (CUBE_SIZE - 1),
                                    sx, sy, sz, center, voxel);
            local_base = __float2int_rz(div_full(__fsub_rn(r_center, r_min), delta_r)) - CACHE_OFFSET;
        }
        __syncthreads();
    }
    auto* sensor_hist = histogram + static_cast<long long>(sensor) * n_bins;
    #pragma unroll 4
    for (int tile = 0; tile < CUBE_SIZE * CUBE_SIZE * CUBE_SIZE / THREADS; ++tile) {
        int ix, iy, iz, source;
        if constexpr (MODE == 0) {
            source = (group * 4 + tile) * THREADS + lane;
            ix = source / (grid_size * grid_size);
            int rem = source - ix * grid_size * grid_size;
            iy = rem / grid_size;
            iz = rem - iy * grid_size;
        } else {
            int local_source = tile * THREADS + lane;
            ix = cube_x + local_source / (CUBE_SIZE * CUBE_SIZE);
            iy = cube_y + (local_source / CUBE_SIZE) % CUBE_SIZE;
            iz = cube_z + local_source % CUBE_SIZE;
            source = (ix * grid_size + iy) * grid_size + iz;
        }
        float r = radius(static_cast<float>(ix), static_cast<float>(iy), static_cast<float>(iz),
                         sx, sy, sz, center, voxel);
        float pos = div_full(__fsub_rn(r, r_min), delta_r);
        int i0 = __float2int_rz(pos);
        float alpha = __fsub_rn(pos, __int2float_rn(i0));
        float weight = div_full(pc[source], __fadd_rn(r, r));
        float u0 = __fmul_rn(weight, __fsub_rn(1.0f, alpha));
        float u1 = __fmul_rn(weight, alpha);
        auto q0 = static_cast<unsigned long long>(quantize(u0, scale));
        auto q1 = static_cast<unsigned long long>(quantize(u1, scale));
        if (i0 >= 0 && i0 < n_bins - 1) {
            if constexpr (MODE == 2) {
                int j0 = i0 - local_base;
                if (j0 >= 0 && j0 < LOCAL_BINS) atomicAdd(local_hist + ((lane / 32) % HIST_SHARDS) * LOCAL_BINS + j0, q0);
                else atomicAdd(sensor_hist + i0, q0);
                if (j0 + 1 >= 0 && j0 + 1 < LOCAL_BINS) atomicAdd(local_hist + ((lane / 32) % HIST_SHARDS) * LOCAL_BINS + j0 + 1, q1);
                else atomicAdd(sensor_hist + i0 + 1, q1);
            } else {
                atomicAdd(sensor_hist + i0, q0);
                atomicAdd(sensor_hist + i0 + 1, q1);
            }
        }
    }
    if constexpr (MODE == 2) {
        __syncthreads();
        for (int local = lane; local < LOCAL_BINS; local += THREADS) {
            unsigned long long value = 0;
            #pragma unroll
            for (int shard = 0; shard < HIST_SHARDS; ++shard)
                value += local_hist[shard * LOCAL_BINS + local];
            int bin = local_base + local;
            if (value != 0 && bin >= 0 && bin < n_bins) atomicAdd(sensor_hist + bin, value);
        }
    }
}

