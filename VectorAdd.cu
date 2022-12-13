//
// Created by WAE on 2022/12/12.
//
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <ctime>

__global__ void vectorAdd(const int* a, const int* b, int* c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        c[i] = a[i] + b[i];
    }
}

void check_result(const std::vector<int>& c, const std::vector<int> &d){
    std::cout <<"===checking GPU's vector with CPU's vector===\n";
    for(int i = 0; i < c.size(); ++i){
        if(c[i] != d[i]){
            std::cout <<"vector check fail\n";
            return;
        }
    }
    std::cout <<"vector check pass\n";
}

void init_vector(std::vector<int> &v){
    for(auto &a:v){
        a = rand() % 100;
    }
}

void cpu_vectorAdd(const std::vector<int>& a, const std::vector<int> &b, std::vector<int> &d){
    for(int i = 0; i < a.size(); ++i){
        d[i] = a[i] + b[i];
    }
}

void print_vector(std::vector<int> &v){
    for(auto &a:v){
        std::cout<<a;
    }
    std::cout<<"\n";
}

int main() {
    int N = 1 << 16;
    std::size_t size = sizeof(int) * N;
    // init vectors and pointer for the vectors
    int *v_a, *v_b, *v_c;
    std::vector<int> a (N);
    std::vector<int> b (N);
    std::vector<int> c (N);
    std::vector<int> d (N);
    // init vector value
    init_vector(a);
    init_vector(b);
    // allocate gpu memory to store vector
    cudaMalloc(&v_a, size);
    cudaMalloc(&v_b, size);
    cudaMalloc(&v_c, size);
    // copy vector from cpu to gpu
    cudaMemcpy(v_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_b, b.data(), size, cudaMemcpyHostToDevice);
    // thread for calculate vector
    int n_thread = 256;
    int n_block = (N + n_thread - 1) / n_thread;
    // run cuda kernel
    clock_t cuda_start = clock();
    vectorAdd<<<n_thread, n_block>>>(v_a, v_b, v_c, N);
    clock_t cuda_end = clock();
    // copy back to cpu memory
    cudaMemcpy(c.data(), v_c, size, cudaMemcpyDeviceToHost);
    clock_t cpu_start = clock();
    cpu_vectorAdd(a, b, d);
    clock_t cpu_end = clock();
    check_result(c, d);
//    print_vector(c);
    // free gpu memory
    cudaFree(v_a);
    cudaFree(v_b);
    cudaFree(v_c);
    std::cout<<"===Runtime comparison===\n";
    std::cout <<"CUDA Runtime: "<<(double)(cuda_end - cuda_start)/CLOCKS_PER_SEC<<"\n";
    std::cout <<"CPU Runtime: "<<(double)(cpu_end - cpu_start)/CLOCKS_PER_SEC<<"\n";
    return 0;
}
