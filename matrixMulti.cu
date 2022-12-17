//
// Created by WAE on 2022/12/14.
//
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

__global__ void matrixMulti (const int* a, const int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * N + col] = 0;
    if((row < N) && (col < N)) {
        for (int k = 0; k < N; ++k) {
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}

void check_result(const int *c, const int *d){
    std::cout <<"===checking GPU's vector with CPU's vector===\n";
    for(int i = 0; i < sizeof(c); ++i){
        if(c[i] != d[i]){
            std::cout <<"vector check fail\n";
            return;
        }
    }
    std::cout <<"vector check pass\n";
}

void init_vector(int *v){
    int n = sizeof(v);
    for(int i = 0;i < n;++i){
        v[i] = rand() % 100;
    }
}

void cpuMatrixMulti(const int *a, const int *b, int *d, int N){
    for(int i = 0; i < N; ++i){
        for(int j = 0;j < N; ++j){
            int tmp = 0;
            for(int k = 0; k < N;++k){
                tmp += a[i * N + k] * b[k * N + j];
            }
            d[i * N + j] = tmp;
        }
    }
}

int main() {
    int N = 1 << 10;
    size_t size = sizeof(int) * N * N;
    int *v_a = (int*) std::malloc(size);
    int *v_b = (int*) std::malloc(size);
    int *v_c = (int*) std::malloc(size);
    int *v_d = (int*) std::malloc(size);
    int *c_a, *c_b, *c_c;
    cudaMalloc(&c_a, size);
    cudaMalloc(&c_b, size);
    cudaMalloc(&c_c, size);
    init_vector(v_a);
    init_vector(v_b);
    cudaMemcpy(c_a, v_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_b, v_b, size, cudaMemcpyHostToDevice);
    // thread for calculate vector
    int n_thread = 32;
    int block = N / n_thread;
    dim3 threads(n_thread, n_thread);
    dim3 blocks(block, block);
    // run cuda kernel
    clock_t cuda_start = clock();
    matrixMulti<<<threads, blocks>>>(c_a, c_b, c_c, N);
    clock_t cuda_end = clock();
    cudaMemcpy(v_c, c_c, size, cudaMemcpyDeviceToHost);
    clock_t cpu_start = clock();
    cpuMatrixMulti(v_a, v_b, v_d, N);
    clock_t cpu_end = clock();
    check_result(v_c, v_d);
    // free gpu memory
    cudaFree(c_a);
    cudaFree(c_b);
    cudaFree(c_c);
    std::free(v_a);
    std::free(v_b);
    std::free(v_c);
    std::free(v_d);
    std::cout<<"===Runtime comparison===\n";
    std::cout <<"CUDA Runtime: "<<(double)(cuda_end - cuda_start)/CLOCKS_PER_SEC<<"\n";
    std::cout <<"CPU Runtime: "<<(double)(cpu_end - cpu_start)/CLOCKS_PER_SEC<<"\n";
    return 0;
}