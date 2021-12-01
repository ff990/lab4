#include <stdio.h>

#define BLOCK_SIZE 512
#define MAX_BLOCK_NUM 16


__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE

    __shared__ unsigned int hist_private[4096];

    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
        hist_private[i] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < num_elements)
    {
        atomicAdd(&(hist_private[input[i]]), 1);
        i += stride;
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
        atomicAdd(&(bins[i]), hist_private[i]);
	
	  /*************************************************************************/
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) 
{

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    int blocknum = (num_elements-1)/BLOCK_SIZE+1;
    dim_grid.x = (blocknum > MAX_BLOCK_NUM ? MAX_BLOCK_NUM : blocknum);
    dim_grid.y = dim_grid.z = 1;

    hist_kernel<<<dim_grid, dim_block>>>(input, bins, num_elements, num_bins);

	  /*************************************************************************/

}


