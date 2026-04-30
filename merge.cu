#include <cuda_runtime.h>
#include <climits>
#include "merge.cuh"

device__ int binarySearch(int* A, int sizeA, int* B, int sizeB, int diag) {
    // finding how many elements to take from A ("i")
    // such that:
    //      i + j = diag   (j comes from B)

    // Boundaries for how many elements we can take from A:
    int low = max(0, diag - sizeB);   // Can't take more from B than it has
    int high = min(diag, sizeA);      // Can't take more from A than it has

    // Binary search over valid "i" values
    while (low < high) {
        int mid = (low + high) >> 1;  // Candidate number of elements from A

        // Elements just to the left/right of partition
        int a_left  = (mid > 0) ? A[mid - 1] : INT_MIN;
        int b_right = (diag - mid < sizeB) ? B[diag - mid] : INT_MAX;

        // If A's left element is too big, we took too many from A
        if (a_left > b_right) {
            high = mid;
        } else {
            int b_left  = (diag - mid > 0) ? B[diag - mid - 1] : INT_MIN;
            int a_right = (mid < sizeA) ? A[mid] : INT_MAX;

            // If B's left element is too big, we need more from A
            if (b_left > a_right)
                low = mid + 1;
            else
                // Found correct partition
                return mid;
        }
    }

    // Final valid partition index
    return low;
}

global__ void parallelMergeKernel(
    int* A, int sizeA,   // Sorted input array A
    int* B, int sizeB,   // Sorted input array B
    int* C)              // Output merged array
{
    // Global thread ID across entire grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of elements to merge
    int total = sizeA + sizeB;

    // Total number of threads working on merge
    int numThreads = gridDim.x * blockDim.x;

    // Each thread gets roughly equal number of output elements
    int elemsPerThread = (total + numThreads - 1) / numThreads;

    // Define this thread's portion of the output (diagonal range)
    int diagStart = tid * elemsPerThread;
    int diagEnd   = min(diagStart + elemsPerThread, total);

    // If thread is out of bounds, exit early
    if (diagStart >= total) return;

    // convert output indices into input partitions using merge path

    // Find how many elements of A belong before diagStart
    int aStart = binarySearch(A, sizeA, B, sizeB, diagStart);
    int bStart = diagStart - aStart;  // Remaining elements come from B

    // Same for end of this thread’s chunk
    int aEnd = binarySearch(A, sizeA, B, sizeB, diagEnd);
    int bEnd = diagEnd - aEnd;

    // Now this thread merges:
    // A[aStart : aEnd] with B[bStart : bEnd]

    int i = aStart;     // Pointer into A
    int j = bStart;     // Pointer into B
    int k = diagStart;  // Pointer into output C

    // Standard sequential merge within assigned partition
    while (i < aEnd && j < bEnd) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    // Copy any remaining elements from A
    while (i < aEnd) {
        C[k++] = A[i++];
    }

    // Copy any remaining elements from B
    while (j < bEnd) {
        C[k++] = B[j++];
    }
}

void parallelMerge(int* d_A, int sizeA,
                   int* d_B, int sizeB,
                   int* d_C)
{
    // Total number of elements in output
    int total = sizeA + sizeB;

    // Standard CUDA configuration
    int threadsPerBlock = 256;

    // Number of blocks needed to cover all elements
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    parallelMergeKernel<<<numBlocks, threadsPerBlock>>>(
        d_A, sizeA,
        d_B, sizeB,
        d_C
    );

    // Wait for GPU to finish
    cudaDeviceSynchronize();
}
