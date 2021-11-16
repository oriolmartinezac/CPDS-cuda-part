#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *dev_u, float *dev_uhelp, int N) {

	// TODO: kernel computation
	//...
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
  }
}
