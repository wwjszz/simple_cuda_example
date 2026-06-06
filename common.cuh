//
// Created by admin on 2026/4/8.
//

#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <cassert>


#define CEIL_DIV(M, N) (((M) + (N) - 1) /  (N))

#define CHECK(call)\
{\
    const cudaError_t error=call; \
    if(error!=cudaSuccess) { \
        std::stringstream ss; \
        ss << "ERROR: " << __FILE__ << ":" << __LINE__ << "."; \
        ss << "code:" << error << ",reason:" << cudaGetErrorString(error); \
        throw std::runtime_error(ss.str()); \
    }\
}

#define TEST_KERNEL(KERNEL_NAME, GRID_DIM, BLOCK_DIM,  ...)                 \
do {                                                                           \
    cpu_time_scope cpu_time(#KERNEL_NAME);                                  \
    KERNEL_NAME<<<GRID_DIM, BLOCK_DIM>>>(__VA_ARGS__);                      \
    cudaDeviceSynchronize();                                                \
} while(0)
#define CHECK_RESULT(KERNEL_NAME, ...) check_result(#KERNEL_NAME, __VA_ARGS__)

#define OFFSET(row, col, len) ((row) * (len) + (col))
#define FETCH_FLOAT4(address) (*reinterpret_cast<float4 *>((address)))

struct cpu_time_scope {
    using clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    cpu_time_scope(std::string name) : name_(std::move(name)) {
        start_time = clock::now();
    }

    ~cpu_time_scope() {
        const auto end_time = clock::now();
        const ms duration = end_time - start_time;
        std::cout << name_ << " elapsed: " << duration.count() << " ms" << std::endl;
    }

    std::chrono::steady_clock::time_point start_time;
    std::string name_;
};

struct gpu_time_scope {
    explicit gpu_time_scope(std::string name) : name_(std::move(name)) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }

    ~gpu_time_scope() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_, stop_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        std::cout << name_ << " elapsed: " << elapsed_ms << " ms" << std::endl;
    }

    cudaEvent_t start_, stop_;
    std::string name_;
};

struct random_generator {
public:
    static float random_float() {
        return dist_float(gen);
    }

    static int random_int() {
        return dist_int(gen);
    }

    template<class T>
    static T random() {
        if (std::is_floating_point_v<T>) {
            return random_float();
        } else {
            return random_int();
        }
    }

private:
    static std::random_device rd;
    static std::mt19937 gen;
    static std::uniform_real_distribution<float> dist_float;
    static std::uniform_int_distribution<int> dist_int;
};

std::random_device random_generator::rd;
std::mt19937 random_generator::gen(rd());
std::uniform_real_distribution<float> random_generator::dist_float(0.0f, 65.535f);
std::uniform_int_distribution<int> random_generator::dist_int(0, 255);

template<class T>
concept arithmetic = std::is_arithmetic_v<T>;

enum class device_type {
    HOST, DEVICE
};

template<arithmetic T>
class cuda_buffer {
public:
    explicit cuda_buffer(const size_t size, device_type type = device_type::HOST) : data_(nullptr), size_(size), type_(type) {
        malloc(size);
    }

    ~cuda_buffer() {
        free();
    }

    T *data() {
        return data_;
    }

    [[nodiscard]] size_t size() const {
        return size_;
    }

    void malloc(size_t size) {
        if (data_ != nullptr) {
            free();
        }

        if (type_ == device_type::HOST) {
            data_ = static_cast<T *>(::malloc(size * sizeof(T)));
        } else {
            CHECK(cudaMalloc(&data_, size * sizeof(T)));
        }
        size_ = size;
    }

    void free() {
        if (data_ != nullptr) {
            if (type_ == device_type::HOST) {
                ::free(data_);
            } else {
                cudaFree(data_);
            }
        }
    }

    void initial_data() {
        for (size_t i = 0; i < size_; i++) {
            data_[i] = random_generator::random<T>();
        }
    }

    void copy_to_host(cuda_buffer<T> &other) {
        assert(other.type_ == device_type::HOST);
        if (type_ == device_type::HOST) {
            CHECK(cudaMemcpy(other.data(), data_, size_ * sizeof(T), cudaMemcpyHostToHost))
        } else {
            CHECK(cudaMemcpy(other.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost))
        }
    }

    void copy_to_device(cuda_buffer<T> &other) {
        assert(other.type_ == device_type::DEVICE);
        if (type_ == device_type::DEVICE) {
            CHECK(cudaMemcpy(other.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice))
        } else {
            CHECK(cudaMemcpy(other.data(), data_, size_ * sizeof(T), cudaMemcpyHostToDevice))
        }
    }

    void copy_to(cuda_buffer<T> &other) {
        if (other.type_ == device_type::HOST) {
            copy_to_host(other);
        } else {
            copy_to_device(other);
        }
    }

    T &operator[](size_t index) {
        return data_[index];
    }

    const T &operator[](size_t index) const {
        return data_[index];
    }

private:
    T *data_{nullptr};
    size_t size_{0};
    const device_type type_{device_type::HOST};
};

// ============================================================
// 基类：管理裸指针 + 大小，禁拷贝、可移动
// ============================================================
template<arithmetic T>
class buffer_base {
protected:
    T *ptr_ = nullptr;
    size_t size_ = 0;

    buffer_base() = default;

    explicit buffer_base(size_t n) : size_(n) {
    }

public:
    buffer_base(const buffer_base &) = delete;

    buffer_base &operator=(const buffer_base &) = delete;

    buffer_base(buffer_base &&o) noexcept
        : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }

    buffer_base &operator=(buffer_base &&o) noexcept {
        if (this != &o) {
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    virtual void clear() {
        delete[] ptr_;
    }

    void destroy() {
        clear();
        ptr_ = nullptr;
        size_ = 0;
    }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
};

// ============================================================
// host_buffer：可在 CPU 上访问，支持迭代
// ============================================================
template<arithmetic T>
class device_buffer; // 前向声明

template<arithmetic T>
class host_buffer : public buffer_base<T> {
    using buffer_base<T>::ptr_;
    using buffer_base<T>::size_;

public:
    host_buffer() = default;

    explicit host_buffer(size_t n) : buffer_base<T>(n) {
        ptr_ = new T[n](); // value-init = 0
    }

    ~host_buffer() { delete[] ptr_; }

    host_buffer(host_buffer &&o) noexcept = default;

    host_buffer &operator=(host_buffer &&o) noexcept = default;

    // 元素访问
    T &operator[](size_t i) { return ptr_[i]; }
    const T &operator[](size_t i) const { return ptr_[i]; }

    // === 迭代器接口，支持 range-based for ===
    T *begin() { return ptr_; }
    T *end() { return ptr_ + size_; }
    const T *begin() const { return ptr_; }
    const T *end() const { return ptr_ + size_; }
    const T *cbegin() const { return ptr_; }
    const T *cend() const { return ptr_ + size_; }

    // 初始化随机数据
    void initial_data(unsigned seed = 42) {
        std::mt19937 gen(seed);
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(T(0), T(1));
            for (auto &v: *this) v = dist(gen);
        } else {
            std::uniform_int_distribution<T> dist(T(0), T(100));
            for (auto &v: *this) v = dist(gen);
        }
    }

    // === host → host ===
    void copy_to_host(host_buffer<T> &dst) const {
        if (dst.size() != size_)
            throw std::runtime_error("size mismatch in copy_to_host (h2h)");
        // 用 cudaMemcpy 也行（cudaMemcpyHostToHost），但 std::memcpy 更直接
        std::memcpy(dst.data(), ptr_, size_ * sizeof(T));
    }

    // === host → device ===
    void copy_to_device(device_buffer<T> &dst) const {
        if (dst.size() != size_)
            throw std::runtime_error("size mismatch in copy_to_device (h2d)");
        CHECK(cudaMemcpy(dst.data(), ptr_,
            size_ * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    device_buffer<T> gen_device() const {
        device_buffer<T> h(size_);
        copy_to_device(h);
        return h;
    }
};

// ============================================================
// device_buffer：仅在 GPU 上有效，不提供 begin/end / operator[]
// ============================================================
template<arithmetic T>
class device_buffer : public buffer_base<T> {
    using buffer_base<T>::ptr_;
    using buffer_base<T>::size_;

public:
    device_buffer() = default;

    explicit device_buffer(size_t n) : buffer_base<T>(n) {
        CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        CHECK(cudaMemset(ptr_, 0, n * sizeof(T)));
    }

    ~device_buffer() {
        if (ptr_) cudaFree(ptr_);
    }

    device_buffer(device_buffer &&) noexcept = default;

    device_buffer &operator=(device_buffer &&) noexcept = default;

    void clear() override {
        if (ptr_) cudaFree(ptr_);
    }

    // device → host
    void copy_to_host(host_buffer<T> &dst) const {
        if (dst.size() != size_)
            throw std::runtime_error("size mismatch in copy_to_host");
        CHECK(cudaMemcpy(dst.data(), ptr_,
            size_ * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    // device → device
    void copy_to_device(device_buffer<T> &dst) const {
        if (dst.size() != size_)
            throw std::runtime_error("size mismatch in copy_to_device");
        CHECK(cudaMemcpy(dst.data(), ptr_,
            size_ * sizeof(T),
            cudaMemcpyDeviceToDevice));
    }

    host_buffer<T> gen_host() {
        host_buffer<T> h(size_);
        copy_to_host(h);
        return h;
    }
};


#endif //CUDA_COMMON_CUH
