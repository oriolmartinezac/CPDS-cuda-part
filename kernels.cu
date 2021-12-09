#include <math.h>
#include <float.h>
#include <cuda.h>

//#define SIZEWRAP 256; //Constant as the size of the wraps

__global__ void gpu_Heat(float *dev_u, float *dev_uhelp, float *dev_res, int N)
{
	// TODO: kernel computation
	//...
  float partial_res;

  unsigned int x = threadIdx.x+1 + blockIdx.x*blockDim.x;
  unsigned int y = threadIdx.y+1 + blockIdx.y*blockDim.y;
  unsigned int my_index = y*N+x;

  //dev_res[(y-1)*N+(x-1)] = 0.0;
  //__syncthreads();

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

__device__ void warpReduce(volatile float *sdata, int tid){
  sdata[tid] += sdata[tid+32];
  sdata[tid] += sdata[tid+16];
  sdata[tid] += sdata[tid+8];
  sdata[tid] += sdata[tid+4];
  sdata[tid] += sdata[tid+2];
  sdata[tid] += sdata[tid+1];
}

__device__ void warpReduce2(volatile float *sdata, unsigned int tid, unsigned int sizewrap){
  if(sizewrap >= 64)sdata[tid] += sdata[tid+32];
  if(sizewrap >= 32)sdata[tid] += sdata[tid+16];
  if(sizewrap >= 16)sdata[tid] += sdata[tid+8];
  if(sizewrap >= 8)sdata[tid] += sdata[tid+4];
  if(sizewrap >= 4)sdata[tid] += sdata[tid+2];
  if(sizewrap >= 2)sdata[tid] += sdata[tid+1];
}

__global__ void gpu_reduce_2(float *dev_res, float *dev_final_res)
{
  extern __shared__ float s_res[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x *(blockDim.x*2) + tid;

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

__global__ void gpu_reduce_3(float *dev_res, float *dev_final_res)
{
  extern __shared__ float s_res[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x *(blockDim.x*2) + tid;
  //unsigned int i = blockIdx.x*blockDim.x + tid;

  //s_res[tid] = dev_res[i];
  s_res[tid] = dev_res[i] + dev_res[i+blockDim.x];
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s>32; s>>=1)
  {
    if(tid<s)
    {
      s_res[tid] += s_res[tid+s];
      __syncthreads();
    }

  }

  if(tid < 32) warpReduce(s_res, tid);

  if (tid == 0) dev_final_res[blockIdx.x] = s_res[0];

}

__global__ void gpu_reduce_4(float *dev_res, float *dev_final_res, unsigned int sizewrap)
{
  extern __shared__ float s_res[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x *(sizewrap*2) + tid;

  //s_res[tid] = dev_res[i] + dev_res[i+blockDim.x];
  s_res[tid] = dev_res[i] + dev_res[i+sizewrap/2];
  __syncthreads();
/*
  for (unsigned int s=blockDim.x/2; s>sizewrap; s>>=1) {
    if (tid < s)
    s_res[tid] += s_res[tid + s];
    __syncthreads();
  }
*/
   if(sizewrap >= 512) {
     if(tid < 256) {
       s_res[tid] += s_res[tid+256];
     }
     __syncthreads();
   }
   if(sizewrap >= 256) {
     if(tid < 128) {
       s_res[tid] += s_res[tid+128];
     }
     __syncthreads();
   }
   if(sizewrap >= 128) {
     if(tid < 64) {
       s_res[tid] += s_res[tid+64];
     }
     __syncthreads();
   }

  if(tid < 32) warpReduce2(s_res, tid, sizewrap);

  if(tid == 0) dev_final_res[blockIdx.x] = s_res[0];

}

__global__ void gpu_reduce_5(float *dev_res, float *dev_final_res, unsigned int sizewrap, unsigned int n)
{
  extern __shared__ float s_res[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x *(sizewrap*2) + tid;
  unsigned int gridSize = sizewrap*2*gridDim.x; //gridDim = 64

  s_res[tid] = 0;

  while(i < n) {
    s_res[tid] += dev_res[i] + dev_res[i+sizewrap];
    i += gridSize;
  }
  __syncthreads();

   if(sizewrap >= 512) {
     if(tid < 256) {
       s_res[tid] += s_res[tid+256];
     }
     __syncthreads();
   }
   if(sizewrap >= 256) {
     if(tid < 128) {
       s_res[tid] += s_res[tid+128];
     }
     __syncthreads();
   }
   if(sizewrap >= 128) {
     if(tid < 64) {
       s_res[tid] += s_res[tid+64];
     }
     __syncthreads();
   }

  if(tid < 32) warpReduce2(s_res, tid, sizewrap);

  if(tid == 0) dev_final_res[blockIdx.x] = s_res[0];

}
/*
template <unsigned int blockSize>
__global__ void gpu_reduce_4(float *dev_res, float *dev_final_res, unsigned int n)
{
  extern __shared__ float s_res[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x *(blockSize*2) + tid;

  s_res[tid] = dev_res[i];
  __syncthreads();

  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      s_res[tid] += s_res[tid+256];
      __syncthreads();
    }
  }
  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      s_res[tid] += s_res[tid+128];
      __syncthreads();
    }
  }
  if (blockSize >= 128)
  {
    if (tid < 64)
    {
      s_res[tid] += s_res[tid+64];
      __syncthreads();
    }
  }

  if(tid < 32) warpReduce(s_res, tid);
    //dev_final_res[blockIdx.x] = s_res[0];
  if (tid == 0) dev_final_res[blockIdx.x] = s_res[0];
}
*/
