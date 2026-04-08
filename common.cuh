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

template<arithmetic T>
class cuda_buffer {
public:
    enum class device_type {
        HOST, DEVICE
    };

    explicit cuda_buffer(const size_t size, device_type type = device_type::HOST) : data_(nullptr), size_(size), type_(type) {
        malloc(size);
    }

    ~cuda_buffer() {
        cudaFree(data_);
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
            data_ = malloc(size * sizeof(T));
        } else {
            CHECK(cudaMalloc(&data_, size * sizeof(T)));
        }
        size_ = size;
    }

    void free() {
        if (data_ != nullptr) {
            if (type_ == device_type::HOST) {
                free(data_);
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

    T& operator[](size_t index) {
        return data_[index];
    }

    const T& operator[](size_t index) const {
        return data_[index];
    }

private:
    T *data_{nullptr};
    size_t size_{0};
    const device_type type_{device_type::HOST};
};


#endif //CUDA_COMMON_CUH
