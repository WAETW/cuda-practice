//
// Created by WAE on 2022/12/14.
//
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <cstdlib>

__global__ void vectorAdd(const int* a, const int* b, int* c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        c[i] = a[i] + b[i];
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

void cpu_vectorAdd(const int *a, const int *b, int *d){
    for(int i = 0; i < sizeof(a); ++i){
        d[i] = a[i] + b[i];
    }
}

void print_vector(const int *v){
    for(int i = 0;i < sizeof(v);++i){
        std::cout<<v[i];
    }
    std::cout<<"\n";
}

int main() {
    int N = 1 << 16;
    std::size_t size = sizeof(int) * N;
    int device_id = cudaGetDevice(&device_id);
    // init vectors and pointer for the vectors
    int *v_a, *v_b, *v_c;
    int *v_d = (int *) std::malloc(size);
    // allocate gpu memory to store vector
    cudaMallocManaged(&v_a, size);
    cudaMallocManaged(&v_b, size);
    cudaMallocManaged(&v_c, size);
    // prefetch data
    cudaMemAdvise(v_a, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(v_b, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(v_c, size, device_id);
    // init vector value
    init_vector(v_a);
    init_vector(v_b);
    // prefetch vector a, b to gpu
    cudaMemAdvise(v_a, size, cudaMemAdviseSetReadMostly, device_id);
    cudaMemAdvise(v_b, size, cudaMemAdviseSetReadMostly, device_id);
    cudaMemPrefetchAsync(v_a, size, device_id);
    cudaMemPrefetchAsync(v_b, size, device_id);
    // thread for calculate vector
    int n_thread = 256;
    int n_block = (N + n_thread - 1) / n_thread;
    // run cuda kernel
    clock_t cuda_start = clock();
    vectorAdd<<<n_thread, n_block>>>(v_a, v_b, v_c, N);
    clock_t cuda_end = clock();
    // copy back to cpu memory
    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(v_a, size, cudaCpuDeviceId);
    cudaMemPrefetchAsync(v_b, size, cudaCpuDeviceId);
    cudaMemPrefetchAsync(v_c, size, cudaCpuDeviceId);
    clock_t cpu_start = clock();
    cpu_vectorAdd(v_a, v_b, v_d);
    clock_t cpu_end = clock();
    check_result(v_c, v_d);
    // free gpu memory
    cudaFree(v_a);
    cudaFree(v_b);
    cudaFree(v_c);
    std::free(v_d);
    std::cout<<"===Runtime comparison===\n";
    std::cout <<"CUDA Runtime: "<<(double)(cuda_end - cuda_start)/CLOCKS_PER_SEC<<"\n";
    std::cout <<"CPU Runtime: "<<(double)(cpu_end - cpu_start)/CLOCKS_PER_SEC<<"\n";
    return 0;
}