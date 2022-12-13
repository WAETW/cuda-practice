#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel() {
    std::printf("Hello world!\n");
}

int main() {
    kernel<<<1,1>>>();
    if(cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    return 0;
}
