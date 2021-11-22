#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat(float *dev_u, float *dev_uhelp, float *dev_res, int N)
{
	// TODO: kernel computation
	//...

  float partial_res;

  unsigned int x = threadIdx.x+1 + blockIdx.x*blockDim.x;
  unsigned int y = threadIdx.y+1 + blockIdx.y*blockDim.y;
  unsigned int my_index = y*N+x;

  if (x <= min((blockIdx.x+1)*blockDim.x, N-2) && y <= min((blockIdx.y+1)*blockDim.y, N-2))
  {
    unsigned int left = y*N + x-1;
    unsigned int right = y*N + x+1;
    unsigned int top = (y-1)*N + x;
    unsigned int bot = (y+1)*N + x;

    dev_uhelp[my_index] = 0.25*(dev_u[left]+dev_u[right]+dev_u[top]+dev_u[bot]);
    partial_res = dev_uhelp[my_index] - dev_u[my_index];
    dev_res[my_index] = partial_res*partial_res;
  }
}


/*
__global__ void gpu_Heat_correct(float *dev_u, float *dev_uhelp, float *dev_res, float *dev_final_res int N)
{
	// TODO: kernel computation
	//...

  extern __shared__ float s_res[];

  unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  float partial_res;

  unsigned int x = threadIdx.x+1 + blockIdx.x*blockDim.x;
  unsigned int y = threadIdx.y+1 + blockIdx.y*blockDim.y;
  unsigned int my_index = y*N+x;

  if (x <= min((blockIdx.x+1)*blockDim.x, N-2) && y <= min((blockIdx.y+1)*blockDim.y, N-2))
  {
    unsigned int left = y*N + x-1;
    unsigned int right = y*N + x+1;
    unsigned int top = (y-1)*N + x;
    unsigned int bot = (y+1)*N + x;

    dev_uhelp[my_index] = 0.25*(dev_u[left]+dev_u[right]+dev_u[top]+dev_u[bot]);
    partial_res = dev_uhelp[my_index] - dev_u[my_index];
    s_res[my_index] = partial_res*partial_res;
  }

  __syncthreads();

  for (unsigned int s = blockDim.x/2+blockDim.y/2; s>0; s>>1)
  {
    if(my_index< s)
    {
      s_res[my_index] += s_res[my_index+s];
      __syncthreads();
    }
  }

  if(tid == 0 || threadIdx.y == 0)
    dev_final_res[blockIdx.x+blockIdx.y] = s_res[0];

}
*/

__global__ void gpu_reduce(float *dev_res, float *dev_final_res)
{
  extern __shared__ float s_res[];

  unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  s_res[tid] = dev_res[i];
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s>0; s>>=1)
  {
    if(tid< s)
    {
      s_res[tid] += s_res[tid+s];
      __syncthreads();
    }
  }

  if(tid == 0)
    dev_final_res[blockIdx.x] = s_res[0];

}

__global__ void gpu_reduce_2(float *dev_res, float *dev_final_res)
{
  extern __shared__ float s_res[];

  unsigned int i = blockIdx.x *(blockDim.x*2) + threadIdx.x;
  unsigned int tid = threadIdx.x;

  s_res[tid] = dev_res[i] + dev_res[i+blockDim.x];
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s>0; s>>=1)
  {
    if(tid< s)
    {
      s_res[tid] += s_res[tid+s];
      __syncthreads();
    }
  }

  if(tid == 0)
    dev_final_res[blockIdx.x] = s_res[0];

}
