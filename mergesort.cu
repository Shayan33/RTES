//MergeSort - Shayan Alipour

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <Windows.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <helper_cuda.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"





long readList(long**);


void mergesort(long*, long, dim3, dim3);

__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);


#define min(a, b) (a < b ? a : b)

int main() {

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

	std::cout << "\nthreadsPerBlock:"
                  << "\n  x: " << threadsPerBlock.x
                  << "\n\nblocksPerGrid:"
                  << "\n  x:" << blocksPerGrid.x
                  << "\n\n total threads: " 
                  << threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                     blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z
                  << "\n\n";

    long* data;
    long size = readList(&data);
	std::cout<<"\nthe data is:"<<data<<"\n";

    if (!size) return -1;

    std::cout << "sorting " << size << " numbers\n\n";

    // merge-sort the data
    mergesort(data, size, threadsPerBlock, blocksPerGrid);


    // Print out the list
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << '\n';
    } 
}

void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
    checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(long)));
    std::cout << "cudaMalloc device lists...\n";

    // Copy from our input list into the first array
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
    std::cout << "cudaMemcpy list to device:...\n";
 
    
    // Copy the thread / block info to the GPU as well
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));
    std::cout << "cudaMalloc device thread data:...\n";

    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));
    std::cout << "cudaMemcpy thread data to device:...\n";

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

      
    std::cout << "mergeSort - width: " << width 
              << ", slices: " << slices 
              << ", nThreads: " << nThreads << '\n';
     

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
		std::cout << "call mergesort kernel:...\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    // Get the list back from the GPU
    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));
	std::cout << "cudaMemcpy list back to host:...\n";
    
    
    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
	std::cout << "cudaFree:...\n";
}


// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

// Perform a full mergesort on our section of the data.
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// read data into a minimal linked list
typedef struct {
    int v;
    void* next;
} LinkNode;

// helper function for reading numbers from stdin
long readList(long** list) {

	std::cout<<"Enter Values:\n";
    long v, size = 0;
    LinkNode* node = 0;
    LinkNode* first = 0;
    while (std::cin >> v) {
        LinkNode* next = new LinkNode();
        next->v = v;
        if (node)
            node->next = next;
        else 
            first = next;
        node = next;
        size++;
    }

    if (size) {
        *list = new long[size]; 
        LinkNode* node = first;
        long i = 0;
        while (node) {
            (*list)[i++] = node->v;
            node = (LinkNode*) node->next;
        }
    }

    std::cout << "read stdin:...\n";
	std::cout<<"\nthe size is:"<< size<<"\n";
    return size;
}
