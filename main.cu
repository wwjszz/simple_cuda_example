#include "common.cuh"


template<arithmetic T>
T cpu_reduction(cuda_buffer<T> &buf) {
    T cpu_sum = 0;
    for (int i = 0; i < buf.size(); i++)
        cpu_sum += buf[i];
    return cpu_sum;
}

template<arithmetic T>
__global__ void warmup(cuda_buffer<T> &in_buf, cuda_buffer<T> &out_buf) {
    size_t n = in_buf.size();
    T *be_buf = in_buf.data() + blockDim.x * blockIdx.x;
    size_t tid = threadIdx.x;

    if (tid >= n) return;

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

    {
        cpu_time_scope cpu_time("cpu reduce");
        int cpu_sum = cpu_reduction(tmp);
        std::cout << "CPU sum: " << cpu_sum << std::endl;
    }

    {
        CHECK(cudaDeviceSynchronize());
        cpu_time_scope cpu_time("warmup");
        warmup<<<grid, block>>>(idata_host, odata_host);
    }


    return 0;
}
