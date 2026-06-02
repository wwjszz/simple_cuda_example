#include "common.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) /  (N))

constexpr int M = 1 << 5, N = 1 << 4, K = 1 << 3;

template<arithmetic T>
void cpu_sgema(cuda_buffer<T> &bufA, cuda_buffer<T> &bufB, cuda_buffer<T> &bufC, int M, int N, int K) {
    for (int i = 0; i < M ; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bufA[i * K + k] * bufB[k * N + j];
            }
            bufC[i * N + j] = sum;
        }
    }
}

template<arithmetic T>
void check_result(std::string_view str, cuda_buffer<T> &odata_host, cuda_buffer<T> &odata_dev, cuda_buffer<T> &odata_tmp) {
    odata_dev.copy_to_host(odata_host);
    assert(odata_tmp.size() == odata_host.size());
    int sz = odata_tmp.size();
    for (int i = 0; i < sz; ++i)
        assert(odata_host[i] == odata_tmp[i]);
}

__global__ void sgema_naive_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}

// TODO: block_x and block_y not equal ???
template<size_t BLOCK_SIZE>
__global__ void sgema_block_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];
    int thread_row = threadIdx.x;
    int thread_col = threadIdx.y;

    int block_offset_row = blockIdx.y * blockDim.y;
    int block_offset_col = blockIdx.x * blockDim.x;

    A += block_offset_row * BLOCK_SIZE * K;
    B += block_offset_col * BLOCK_SIZE;
    C += block_offset_row * BLOCK_SIZE * N + block_offset_col * BLOCK_SIZE;

    // MxK, KxN
    float tmp = 0.0f;
    for (int idx = 0; idx < K; idx += BLOCK_SIZE) {
        As[thread_row * BLOCK_SIZE + thread_col] = A[thread_row *  K + thread_col];
        As[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            // As[x][y] Bs[y][x]
            tmp += As[thread_row * BLOCK_SIZE + i] * Bs[thread_col + i * BLOCK_SIZE];
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }

    C[thread_row * N + thread_col] = tmp;
}

int main() {
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int blockX = 32, blockY = 32;

    dim3 blockDim(blockX, blockY);
    dim3 gridDim(CEIL_DIV(M, blockX), CEIL_DIV(N, blockY));

    cuda_buffer<float> idata_a_host(sizeA);
    cuda_buffer<float> idata_b_host(sizeB);
    cuda_buffer<float> odata_tmp(sizeC);
    cuda_buffer<float> odata_host(sizeC);

    cpu_sgema(idata_a_host, idata_b_host, odata_tmp, M, N, K);

    cuda_buffer<float> idata_a_dev(sizeA, device_type::DEVICE);
    cuda_buffer<float> idata_b_dev(sizeB, device_type::DEVICE);
    cuda_buffer<float> odata_dev(sizeC, device_type::DEVICE);

    idata_a_host.initial_data();
    idata_b_host.initial_data();

    idata_a_host.copy_to_device(idata_a_dev);
    idata_b_host.copy_to_device(idata_b_dev);
    {
        cpu_time_scope cpu_time("sgema_naive_kernel");
        sgema_naive_kernel<<<gridDim, blockDim>>>(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("sgema_naive_kernel", odata_host, odata_dev, odata_tmp);

    {
        cpu_time_scope cpu_time("sgema_block_kernel");
        sgema_block_kernel<32><<<gridDim, blockDim>>>(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("sgema_block_kernel", odata_host, odata_dev, odata_tmp);

    std::cout << "test success" << std::endl;
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
