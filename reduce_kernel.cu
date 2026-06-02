#include "common.cuh"


template<arithmetic T>
T cpu_reduction(cuda_buffer<T> &buf) {
    T cpu_sum = 0;
    for (int i = 0; i < buf.size(); i++)
        cpu_sum += buf[i];
    return cpu_sum;
}

template<arithmetic T>
__global__ void warmup(T* inbuf, T* out_buf, size_t n) {
    T *be_buf = inbuf + blockDim.x * blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stripe = 1; stripe < blockDim.x; stripe <<= 1) {
        if ((tid % (2 * stripe)) == 0) {
            be_buf[tid] += be_buf[tid + stripe];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_buf[blockIdx.x] = be_buf[0];
    }
}

template<arithmetic T>
__global__ void reduce_neighbored_less(T* inbuf, T* out_buf, size_t n) {
    T *be_buf = inbuf + blockDim.x * blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;

    if (idx > n) return;

    for (int stripe = 1; stripe < blockDim.x; stripe <<= 1) {
        int index = 2 * tid * stripe;
        if (index < blockDim.x) {
            be_buf[index] += be_buf[index + stripe];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_buf[blockIdx.x] = be_buf[0];
    }
}

template<arithmetic T>
__global__ void reduce_interleaved(T* inbuf, T* out_buf, size_t n) {
    T *be_buf = inbuf + blockDim.x * blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;

    if (idx > n) return;

    for (int stripe = blockDim.x >> 1; stripe > 0; stripe >>= 1) {
        if (tid < stripe) {
            be_buf[tid] += be_buf[tid + stripe];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_buf[blockIdx.x] = be_buf[0];
    }
}

template<arithmetic T, std::size_t N>
__global__ void reduce_unroll(T* inbuf, T* out_buf, size_t n) {
    T *be_buf = inbuf + blockDim.x * blockIdx.x * N;
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x * N;

    if (idx > n) return;

    if (tid < blockDim.x) {
        for (int i = 0; i < N; i++) {
            be_buf[tid] += be_buf[tid + blockDim.x * (i + 1)];
        }
    }
    __syncthreads();

    for (int stripe = blockDim.x >> 1; stripe > 32; stripe >>= 1) {
        if (tid < stripe) {
            be_buf[tid] += be_buf[tid + stripe];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile T* vmem = be_buf;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) {
        out_buf[blockIdx.x] = be_buf[0];
    }
}

template<arithmetic T>
__global__ void reduce_unroll2(T* inbuf, T* out_buf, size_t n) {
    T *be_buf = inbuf + blockDim.x * blockIdx.x * 2;
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x * 2;

    if (idx > n) return;

    if (tid < blockDim.x) {
        be_buf[tid] += be_buf[tid + blockDim.x];
    }
    __syncthreads();

    for (int stripe = blockDim.x >> 1; stripe > 0; stripe >>= 1) {
        if (tid < stripe) {
            be_buf[tid] += be_buf[tid + stripe];
        }
        __syncthreads();
    }


    if (tid == 0) {
        out_buf[blockIdx.x] = be_buf[0];
    }
}


template<arithmetic T>
void check_result(std::string str,cuda_buffer<T> &odata_dev, cuda_buffer<T> &odata_host, T sum, size_t n) {
    odata_dev.copy_to_host(odata_host);
    T gpu_sum = 0;
    for (int i = 0; i < n; i++) {
        gpu_sum += odata_host[i];
    }
    // std::cout << str << " sum: " << gpu_sum << std::endl;
    if (gpu_sum == sum) {
        std::cout << str << " test success!" << std::endl;
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, dev));
    cudaSetDevice(dev);

    int size = 1 << 24;
    std::cout << "array size: " << size << std::endl;

    //execution configuration
    int blocksize = 1024;
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    std::cout << "grid: " << grid.x << " block: " << block.x << std::endl;

    cuda_buffer<int> idata_host(size);
    cuda_buffer<int> odata_host(grid.x);
    cuda_buffer<int> tmp(size);

    idata_host.initial_data();
    idata_host.copy_to_host(tmp);

    cuda_buffer<int> idata_dev(size, device_type::DEVICE);
    cuda_buffer<int> odata_dev(grid.x, device_type::DEVICE);

    int cpu_sum = 0;
    {
        cpu_time_scope cpu_time("cpu reduce");
        cpu_sum = cpu_reduction(tmp);
        std::cout << "CPU sum: " << cpu_sum << std::endl;
    }

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("warmup");
        warmup<<<grid, block>>>(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("warmup", odata_dev, odata_host, cpu_sum, grid.x);

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("reduce_neighbored_less");
        reduce_neighbored_less<<<grid, block>>>(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("reduce_neighbored_less", odata_dev, odata_host, cpu_sum, grid.x);

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("reduce_interleaved");
        reduce_interleaved<<<grid, block>>>(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("reduce_interleaved", odata_dev, odata_host, cpu_sum, grid.x);

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("reduce_unroll2");
        reduce_unroll2<<<grid.x / 2, block>>>(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("reduce_unroll2", odata_dev, odata_host, cpu_sum, grid.x / 2);

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("reduce_unroll<2>");
        reduce_unroll <int, 2> << <grid.x / 2, block>> >(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("reduce_unroll<2>", odata_dev, odata_host, cpu_sum, grid.x / 2);

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("reduce_unroll<4>");
        reduce_unroll <int, 4> << <grid.x / 4, block>> >(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("reduce_unroll<4>", odata_dev, odata_host, cpu_sum, grid.x / 4);

    idata_host.copy_to_device(idata_dev);
    CHECK(cudaDeviceSynchronize());
    {
        cpu_time_scope cpu_time("reduce_unroll<8>");
        reduce_unroll <int, 8> << <grid.x / 8, block>> >(idata_dev.data(), odata_dev.data(), size);
        cudaDeviceSynchronize();
    }
    check_result("reduce_unroll<8>", odata_dev, odata_host, cpu_sum, grid.x / 8);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
