#include "common.cuh"
#include <print>

constexpr size_t ARRAY_SIZE = 1 << 26;

template<arithmetic T>
__host__ T cpu_reduce(host_buffer<T> &idata_host) {
    T sum{};
    for (T i: idata_host) sum += i;
    return sum;
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_native_kernel(T *idata, T *odata, int N) {
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id + thread_id;

    // copy to shared
    if (tid < N) sdata[thread_id] = idata[tid];
    __syncthreads();

    // thread sum
    for (int s = 1; s < block_dim; s <<= 1) {
        if ((thread_id % (s << 1) == 0) && (thread_id + s < block_dim) && (tid + s < N)) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        odata[block_id] = sdata[0];
    }
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_interleave_kernel(T *idata, T *odata, int N) {
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id + thread_id;

    // copy to shared
    if (tid < N) sdata[thread_id] = idata[tid];
    __syncthreads();

    // thread sum
    for (size_t s = 1; s < block_dim; s <<= 1) {
        size_t index = thread_id * s * 2;
        if (index + s < block_dim && (block_dim * block_id + index + s < N)) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        odata[block_id] = sdata[0];
    }
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_bank_conflict_kernel(T *idata, T *odata, int N) {
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id + thread_id;

    // copy to shared
    if (tid < N) sdata[thread_id] = idata[tid];
    __syncthreads();

    // thread sum
    for (size_t s = block_dim >> 1; s > 0; s >>= 1) {
        if (thread_id < s) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        odata[block_id] = sdata[0];
    }
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_idle_kernel(T *idata, T *odata, int N) {
    int stride = 2;
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id * stride + thread_id;

    // copy to shared
    if (tid < N) {
        sdata[thread_id] = idata[tid] + idata[tid + BLOCK_SIZE];
    }
    __syncthreads();

    // thread sum
    for (size_t s = block_dim >> 1; s > 0; s >>= 1) {
        if (thread_id < s) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        odata[block_id] = sdata[0];
    }
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_wrap_unroll_kernel(T *idata, T *odata, int N) {
    int stride = 2;
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id * stride + thread_id;

    // copy to shared
    if (tid < N) {
        sdata[thread_id] = idata[tid] + idata[tid + BLOCK_SIZE];
    }
    __syncthreads();

    // thread sum
    for (size_t s = block_dim >> 1; s > 32; s >>= 1) {
        if (thread_id < s) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    if (thread_id < 32) {
        volatile T* vmem = sdata;
        vmem[thread_id] += vmem[thread_id + 32];
        vmem[thread_id] += vmem[thread_id + 16];
        vmem[thread_id] += vmem[thread_id + 8];
        vmem[thread_id] += vmem[thread_id + 4];
        vmem[thread_id] += vmem[thread_id + 2];
        vmem[thread_id] += vmem[thread_id + 1];
    }

    if (thread_id == 0) {
        odata[block_id] = sdata[0];
    }
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_wrap_unroll_sync_warp_kernel(T *idata, T *odata, int N) {
    int stride = 2;
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id * stride + thread_id;

    // copy to shared
    if (tid < N) {
        sdata[thread_id] = idata[tid] + idata[tid + BLOCK_SIZE];
    }
    __syncthreads();

    // thread sum
    for (size_t s = block_dim >> 1; s > 32; s >>= 1) {
        if (thread_id < s) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    if (thread_id < 32) {
        sdata[thread_id] += sdata[thread_id + 32];
        __syncwarp();
        sdata[thread_id] += sdata[thread_id + 16];
        __syncwarp();
        sdata[thread_id] += sdata[thread_id + 8];
        __syncwarp();
        sdata[thread_id] += sdata[thread_id + 4];
        __syncwarp();
        sdata[thread_id] += sdata[thread_id + 2];
        __syncwarp();
        sdata[thread_id] += sdata[thread_id + 1];
    }

    if (thread_id == 0) {
        odata[block_id] = sdata[0];
    }
}

template<size_t BLOCK_SIZE, arithmetic T>
__global__ void reduce_wrap_unroll_shfl_down_sync_kernel(T *idata, T *odata, int N) {
    int stride = 2;
    __shared__ T sdata[BLOCK_SIZE];

    size_t thread_id = threadIdx.x;
    size_t block_id = blockIdx.x;
    size_t block_dim = blockDim.x;
    size_t tid = block_dim * block_id * stride + thread_id;

    // copy to shared
    if (tid < N) {
        sdata[thread_id] = idata[tid] + idata[tid + BLOCK_SIZE];
    }
    __syncthreads();

    // reduce
    if (block_dim >= 256) {
        if (thread_id < 128) {
            sdata[thread_id] += sdata[thread_id + 128];
        }
        __syncthreads();
    }

    if (block_dim >= 128) {
        if (thread_id < 64) {
            sdata[thread_id] += sdata[thread_id + 64];
        }
        __syncthreads();
    }

    if (thread_id < 32) {
        T val = sdata[thread_id] + sdata[thread_id + 32];
        val +=  __shfl_down_sync(0xffffffff, val, 16);
        val +=  __shfl_down_sync(0xffffffff, val, 8);
        val +=  __shfl_down_sync(0xffffffff, val, 4);
        val +=  __shfl_down_sync(0xffffffff, val, 2);
        val +=  __shfl_down_sync(0xffffffff, val, 1);
        if (thread_id == 0) odata[block_id] = val;
    }
}

template<arithmetic T>
__host__ void check_result(std::string_view str, T cpu_result, device_buffer<T> &idata_device) {
    host_buffer<T> idata_host = idata_device.gen_host();

    T sum{};
    for (T i: idata_host) sum += i;
    std::cout << str << "\n\t > test sum=" << sum << std::endl;

    if (sum != cpu_result) {
        std::cout << "\t > ERROR at " << sum << " ref=" << cpu_result << std::endl;
        return;
    }
    std::cout << "\t > test success" << std::endl;
}


int main() {
    host_buffer<long long> idata_host(ARRAY_SIZE);
    device_buffer<long long> idata_device(ARRAY_SIZE);

    idata_host.initial_data();
    idata_host.copy_to_device(idata_device);

    long long cpu_sum = 0;
    {
        cpu_time_scope cpu_time("cpu_reduce");
        cpu_sum = cpu_reduce(idata_host);
    }

    constexpr int block_size = 256; // 1 << 8
    constexpr int grid_size = CEIL_DIV(ARRAY_SIZE, block_size);

    device_buffer<long long> odata_device(grid_size);

    TEST_KERNEL(reduce_native_kernel<block_size>, grid_size, block_size, idata_device.data(), odata_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_native_kernel<block_size>, cpu_sum, odata_device);

    TEST_KERNEL(reduce_interleave_kernel<block_size>, grid_size, block_size, idata_device.data(), odata_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_interleave_kernel<block_size>, cpu_sum, odata_device);

    TEST_KERNEL(reduce_bank_conflict_kernel<block_size>, grid_size, block_size, idata_device.data(), odata_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_bank_conflict_kernel<block_size>, cpu_sum, odata_device);

    device_buffer<long long> odata_idle_device(grid_size >> 1);
    TEST_KERNEL(reduce_idle_kernel<block_size>, grid_size >> 1, block_size, idata_device.data(), odata_idle_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_idle_kernel<block_size>, cpu_sum, odata_idle_device);

    TEST_KERNEL(reduce_wrap_unroll_kernel<block_size>, grid_size >> 1, block_size, idata_device.data(), odata_idle_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_wrap_unroll_kernel<block_size>, cpu_sum, odata_idle_device);

    TEST_KERNEL(reduce_wrap_unroll_sync_warp_kernel<block_size>, grid_size >> 1, block_size, idata_device.data(), odata_idle_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_wrap_unroll_sync_warp_kernel<block_size>, cpu_sum, odata_idle_device);

    TEST_KERNEL(reduce_wrap_unroll_shfl_down_sync_kernel<block_size>, grid_size >> 1, block_size, idata_device.data(), odata_idle_device.data(), ARRAY_SIZE);
    CHECK_RESULT(reduce_wrap_unroll_shfl_down_sync_kernel<block_size>, cpu_sum, odata_idle_device);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
