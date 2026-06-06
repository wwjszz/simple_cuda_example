#include "common.cuh"

constexpr int M = 1 << 10, N = 1 << 10, K = 1 << 10;  // 4096

template<arithmetic T>
__host__ void cpu_matmul(cuda_buffer<T> &bufA, cuda_buffer<T> &bufB, cuda_buffer<T> &bufC, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
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
__host__ bool check_result(std::string_view str, cuda_buffer<T> &odata_host, cuda_buffer<T> &odata_dev, cuda_buffer<T> &odata_tmp) {
    odata_dev.copy_to_host(odata_host);
    int sz = odata_tmp.size();
    for (int i = 0; i < sz; ++i) {
        float a = odata_host[i];
        float b = odata_tmp[i];
        float diff = std::abs(a - b);
        if (diff > 1) {
            std::cout << str << " ERROR at " << i
                    << " host=" << std::setprecision(9) << a
                    << " ref=" << std::setprecision(9) << b
                    << " diff=" << diff << std::endl;
            return false;
        }
    }
    std::cout << str << " success" << std::endl;
    return true;
}

__global__ void matmul_naive_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}

// TODO: block_x and block_y not equal ???
template<unsigned BLOCK_SIZE>
__global__ void matmul_block_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    A += block_row * BLOCK_SIZE * K;
    B += block_col * BLOCK_SIZE;
    C += block_row * BLOCK_SIZE * N + block_col * BLOCK_SIZE;

    // MxK, KxN
    float tmp = 0.0f;
    for (int idx = 0; idx < K; idx += BLOCK_SIZE) {
        As[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        Bs[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];

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

template<unsigned BM, unsigned BK, unsigned BN, unsigned TM>
__global__ void matmul_thread_tile_1d_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // block
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // BM * BK
    int load_a_row = threadIdx.x / BK;
    int load_a_col = threadIdx.x % BK;

    // BK * BN
    int load_b_row = threadIdx.x / BN;
    int load_b_col = threadIdx.x % BN;

    // (BM / TM) * BN
    int cal_row = threadIdx.x / BN;
    int cal_col = threadIdx.x % BN;

    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    float thread_tile[TM] = {0.0f};
    for (int idx = 0; idx < K; idx += BK) {
        As[load_a_row * BK + load_a_col] = A[load_a_row * K + load_a_col];
        Bs[load_b_row * BN + load_b_col] = B[load_b_row * N + load_b_col];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int i = 0; i < BK; ++i) {
            float b_elem = Bs[i * BN + cal_col];
            for (int j = 0; j < TM; ++j) {
                thread_tile[j] += As[(cal_row * TM + j) * BK + i] * b_elem;
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        C[((cal_row * TM) + i) * N + cal_col] = thread_tile[i];
    }
}

template<unsigned BM, unsigned BK, unsigned BN, unsigned TM, unsigned TN>
__global__ void matmul_thread_tile_2d_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // block
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // BM * BK
    int load_a_row = threadIdx.x / BK;
    int load_a_col = threadIdx.x % BK;

    // BK * BN
    int load_b_row = threadIdx.x / BN;
    int load_b_col = threadIdx.x % BN;

    // (BM / TM) * (BN / TN)
    const unsigned CN = BN / TN;
    unsigned cal_row = threadIdx.x / CN;
    unsigned cal_col = threadIdx.x % CN;

    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    int stride_a = blockDim.x / BK;
    int stride_b = blockDim.x / BN;

    float thread_tile[TM * TN] = {0.0f};
    float back_a[TM] = {0.0f};
    float back_b[TN] = {0.0f};

    for (int idx = 0; idx < K; idx += BK) {
        for (int i = 0; i < BM; i += stride_a) As[(load_a_row + i) * BK + load_a_col] = A[(load_a_row + i) * K + load_a_col];
        for (int i = 0; i < BK; i += stride_b) Bs[(load_b_row + i) * BN + load_b_col] = B[(load_b_row + i) * N + load_b_col];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) back_a[i] = As[(cal_row * TM + i) * BK + k];
            for (int i = 0; i < TN; ++i) back_b[i] = Bs[k * BN + cal_col * TN + i];
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_tile[i * TN + j] += back_a[i] * back_b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            C[(cal_row * TM + i) * N + (cal_col * TN + j)] = thread_tile[i * TN + j];
        }
}

constexpr unsigned VEC_SIZE = sizeof(float4) / sizeof(float);

template<unsigned BM, unsigned BK, unsigned BN, unsigned TM, unsigned TN>
__global__ void matmul_thread_tile_vectorize_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // block
    unsigned block_row = blockIdx.y;
    unsigned block_col = blockIdx.x;

    // BM * BK
    unsigned VEC_A_COL_SIZE = BK / VEC_SIZE;
    unsigned load_a_row = threadIdx.x / VEC_A_COL_SIZE;
    unsigned load_a_col = (threadIdx.x % VEC_A_COL_SIZE) * VEC_SIZE;

    // BK * BN
    unsigned VEC_B_COL_SIZE = BN / VEC_SIZE;
    unsigned load_b_row = threadIdx.x / VEC_B_COL_SIZE;
    unsigned load_b_col = (threadIdx.x % VEC_B_COL_SIZE) * VEC_SIZE;

    // (BM / TM) * (BN / TN)
    const unsigned CN = BN / TN;
    unsigned cal_row = threadIdx.x / CN;
    unsigned cal_col = threadIdx.x % CN;

    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    unsigned stride_a = blockDim.x / VEC_A_COL_SIZE;
    unsigned stride_b = blockDim.x / VEC_B_COL_SIZE;

    float thread_tile[TM * TN] = {0.0f};
    float back_a[TM] = {0.0f};
    float back_b[TN] = {0.0f};

    float cache_a[VEC_SIZE] = {0.0f};

    for (unsigned idx = 0; idx < K; idx += BK) {
        for (unsigned i = 0; i < BM; i += stride_a) {
            FETCH_FLOAT4(cache_a) = FETCH_FLOAT4(&A[OFFSET(load_a_row + i, load_a_col, K)]);
            As[OFFSET(load_a_col, load_a_row + i, BM)] = cache_a[0];
            As[OFFSET(load_a_col + 1, load_a_row + i, BM)] = cache_a[1];
            As[OFFSET(load_a_col + 2, load_a_row + i, BM)] = cache_a[2];
            As[OFFSET(load_a_col + 3, load_a_row + i, BM)] = cache_a[3];
        }
        for (unsigned i = 0; i < BK; i += stride_b) {
            FETCH_FLOAT4(&Bs[OFFSET(load_b_row + i, load_b_col, BN)]) = FETCH_FLOAT4(&B[OFFSET(load_b_row + i, load_b_col, N)]);
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; i += 4) {
                FETCH_FLOAT4(&back_a[i]) = FETCH_FLOAT4(&As[OFFSET(k, cal_row * TM  + i, BM)]);
            }
            for (int i = 0; i < TN; i += 4) {
                FETCH_FLOAT4(&back_b[i]) = FETCH_FLOAT4(&Bs[OFFSET(k, cal_col * TN + i, BN)]);
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_tile[OFFSET(i, j, TN)] += back_a[i] * back_b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4) {
            FETCH_FLOAT4(&C[OFFSET(cal_row * TM + i, cal_col * TN + j, N)]) = FETCH_FLOAT4(&thread_tile[OFFSET(i, j, TN)]);
        }
}

template<unsigned BM, unsigned BK, unsigned BN, unsigned TM, unsigned TN>
__global__ void matmul_thread_tile_double_buffer_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // block
    unsigned block_row = blockIdx.y;
    unsigned block_col = blockIdx.x;

    // BM * BK
    unsigned VEC_A_COL_SIZE = BK / VEC_SIZE;
    unsigned load_a_row = threadIdx.x / VEC_A_COL_SIZE;
    unsigned load_a_col = (threadIdx.x % VEC_A_COL_SIZE) * VEC_SIZE;

    // BK * BN
    unsigned VEC_B_COL_SIZE = BN / VEC_SIZE;
    unsigned load_b_row = threadIdx.x / VEC_B_COL_SIZE;
    unsigned load_b_col = (threadIdx.x % VEC_B_COL_SIZE) * VEC_SIZE;

    // (BM / TM) * (BN / TN)
    const unsigned CN = BN / TN;
    unsigned cal_row = threadIdx.x / CN;
    unsigned cal_col = threadIdx.x % CN;

    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    unsigned stride_a = blockDim.x / VEC_A_COL_SIZE;
    unsigned stride_b = blockDim.x / VEC_B_COL_SIZE;

    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    float thread_tile[TM * TN] = {0.0f};
    float back_a[TM] = {0.0f};
    float back_b[TN] = {0.0f};

    float cache_a[VEC_SIZE] = {0.0f};

    unsigned write_idx = 0;
    for (unsigned idx = 0; idx < K; idx += BK) {
        for (unsigned i = 0; i < BM; i += stride_a) {
            FETCH_FLOAT4(cache_a) = FETCH_FLOAT4(&A[OFFSET(load_a_row + i, load_a_col, K)]);
            As[write_idx][OFFSET(load_a_col, load_a_row + i, BM)] = cache_a[0];
            As[write_idx][OFFSET(load_a_col + 1, load_a_row + i, BM)] = cache_a[1];
            As[write_idx][OFFSET(load_a_col + 2, load_a_row + i, BM)] = cache_a[2];
            As[write_idx][OFFSET(load_a_col + 3, load_a_row + i, BM)] = cache_a[3];
        }
        for (unsigned i = 0; i < BK; i += stride_b) {
            FETCH_FLOAT4(&Bs[write_idx][OFFSET(load_b_row + i, load_b_col, BN)]) = FETCH_FLOAT4(&B[OFFSET(load_b_row + i, load_b_col, N)]);
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; i += 4) {
                FETCH_FLOAT4(&back_a[i]) = FETCH_FLOAT4(&As[write_idx][OFFSET(k, cal_row * TM  + i, BM)]);
            }
            for (int i = 0; i < TN; i += 4) {
                FETCH_FLOAT4(&back_b[i]) = FETCH_FLOAT4(&Bs[write_idx][OFFSET(k, cal_col * TN + i, BN)]);
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_tile[OFFSET(i, j, TN)] += back_a[i] * back_b[j];
                }
            }
        }
        write_idx =  1 - write_idx;
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4) {
            FETCH_FLOAT4(&C[OFFSET(cal_row * TM + i, cal_col * TN + j, N)]) = FETCH_FLOAT4(&thread_tile[OFFSET(i, j, TN)]);
        }
}


void run_matmul_thread_tile_1d_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const unsigned BM = 64;
    const unsigned BN = 64;
    const unsigned BK = 8;
    const unsigned TM = 8;
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block_size((BM * BN) / TM);
    matmul_thread_tile_1d_kernel<BM, BK, BN, TM><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

void run_matmul_thread_tile_2d_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const unsigned BM = 64;
    const unsigned BN = 64;
    const unsigned BK = 8;
    const unsigned TM = 8;
    const unsigned TN = 8;
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block_size((BM * BN) / (TM * TN));
    matmul_thread_tile_2d_kernel<BM, BK, BN, TM, TN><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

void run_matmul_thread_tile_vectorize_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const unsigned BM = 64;
    const unsigned BN = 64;
    const unsigned BK = 8;
    const unsigned TM = 8;
    const unsigned TN = 8;
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block_size((BM * BN) / (TM * TN));
    matmul_thread_tile_vectorize_kernel<BM, BK, BN, TM, TN><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

void run_matmul_thread_tile_double_buffer_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const unsigned BM = 64;
    const unsigned BN = 64;
    const unsigned BK = 8;
    const unsigned TM = 8;
    const unsigned TN = 8;
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block_size((BM * BN) / (TM * TN));
    matmul_thread_tile_double_buffer_kernel<BM, BK, BN, TM, TN><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

int main() {
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int blockX = 32, blockY = 32;

    dim3 blockDim(blockX, blockY);
    dim3 gridDim(CEIL_DIV(N, blockX), CEIL_DIV(M, blockY));

    cuda_buffer<float> idata_a_host(sizeA);
    cuda_buffer<float> idata_b_host(sizeB);
    cuda_buffer<float> odata_tmp(sizeC);
    cuda_buffer<float> odata_host(sizeC);

    cuda_buffer<float> idata_a_dev(sizeA, device_type::DEVICE);
    cuda_buffer<float> idata_b_dev(sizeB, device_type::DEVICE);
    cuda_buffer<float> odata_dev(sizeC, device_type::DEVICE);

    idata_a_host.initial_data();
    idata_b_host.initial_data();

    idata_a_host.copy_to_device(idata_a_dev);
    idata_b_host.copy_to_device(idata_b_dev);

    cpu_matmul(idata_a_host, idata_b_host, odata_tmp, M, N, K);

    {
        gpu_time_scope gpu_time("matmul_naive_kernel");
        matmul_naive_kernel<<<gridDim, blockDim>>>(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("matmul_naive_kernel", odata_host, odata_dev, odata_tmp);

    {
        gpu_time_scope gpu_time("matmul_block_kernel");
        matmul_block_kernel<32><<<gridDim, blockDim>>>(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("matmul_block_kernel", odata_host, odata_dev, odata_tmp);

    {
        gpu_time_scope gpu_time("matmul_thread_tile_1d_kernel");
        run_matmul_thread_tile_1d_kernel(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("matmul_thread_tile_1d_kernel", odata_host, odata_dev, odata_tmp);

    {
        gpu_time_scope gpu_time("matmul_thread_tile_2d_kernel");
        run_matmul_thread_tile_2d_kernel(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("matmul_thread_tile_2d_kernel", odata_host, odata_dev, odata_tmp);

    {
        gpu_time_scope gpu_time("matmul_thread_tile_vectorize_kernel");
        run_matmul_thread_tile_vectorize_kernel(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("matmul_thread_tile_vectorize_kernel", odata_host, odata_dev, odata_tmp);

    {
        gpu_time_scope gpu_time("run_matmul_thread_tile_double_buffer_kernel");
        run_matmul_thread_tile_double_buffer_kernel(idata_a_dev.data(), idata_b_dev.data(), odata_dev.data(), M, N, K);
        cudaDeviceSynchronize();
    }
    check_result("run_matmul_thread_tile_double_buffer_kernel", odata_host, odata_dev, odata_tmp);

    std::cout << "test success" << std::endl;
    // 注意：cudaDeviceReset 已删除，让 cuda_buffer 析构函数正常释放资源
    return EXIT_SUCCESS;
}
