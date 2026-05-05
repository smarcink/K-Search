#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

void launch_elementwise_add_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    __half* __restrict__ z,
    int n,
    cudaStream_t stream
);