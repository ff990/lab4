#include <stdio.h>

__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
	extern __shared__ unsigned int histo_private[];
	unsigned int i=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int stride=blockDim.x*gridDim.x;
	
	for(int k=0;k<(num_bins-1)/blockDim.x+1;++k)
	    if(threadIdx.x+blockDim.x*k<num_bins)
	       histo_private[threadIdx.x+blockDim.x*k]=0;
	__syncthreads();
	
	while(i<num_elements)
	{
		atomicAdd(&(histo_private[input[i]]),1);
		i+=stride;
	}
        __syncthreads();

	for(int m=0;m<(num_bins-1)/blockDim.x+1;++m)
	    if(threadIdx.x+blockDim.x*m<num_bins)
	       atomicAdd(&(bins[threadIdx.x+blockDim.x*m]),histo_private[threadIdx.x+blockDim.x*m]);
	  /*************************************************************************/
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

	  /*************************************************************************/
    //INSERT CODE HERE
        const int BLOCK_SIZE=512;
	histo_kernel<<<(num_elements-1)/BLOCK_SIZE+1,BLOCK_SIZE,num_bins*sizeof(unsigned int)>>>(input,bins,num_elements,num_bins);

	  /*************************************************************************/

}


