#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {

	// TODO: kernel computation
	//...
  //h = u
  //g = uhelp
  /*
  nbx = NB;
  bx = sizex/nbx;
  nby = NB;
  by = sizey/nby;
  for (int ii=0; ii<nbx; ii++)
      for (int jj=0; jj<nby; jj++)
          for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
              for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
             u[ i*sizey     + (j+1) ]+  // right
                   u[ (i-1)*sizey + j     ]+  // top
                   u[ (i+1)*sizey + j     ]); // bottom
                diff = utmp[i*sizey+j] - u[i*sizey + j];
                sum += diff * diff;
        }
  return(sum);
  */

  /*Ejemplo internet*/
  unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int left = y*N + x-1;
  unsigned int right = y*N + x+1;
  unsigned int top = (y-1)*N + x;
  unsigned int bot = (y+1)*N + x;



/*
  unsigned int bx = N/blockDim.x;
  unsigned int by = N/blockDim.y;

  unsigned int my_indexcc = blockDim*blockIdx.x+threadIdx.x;
  unsigned int my_indexc = threadIdx.y+blockDim.y*blockIdx.y + threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int my_index = threadIdx.y*N + threadIdx.x;
  unsigned int my_neighbour_left = threadIdx.y*N + threadIdx.x-1;
  unsigned int my_neighbour_right = threadIdx.y*N + threadIdx.x+1;
  unsigned int my_neighbour_top = (threadIdx.y-1)*N + threadIdx.x;
  unsigned int my_neighbour_bottom = (threadIdx.y+1)*N + threadIdx.x;
*/
 if()
  if ((my_index < N*N) and (threadIdx.x > 0 and threadIdx.y > 0))
    g[my_index] = 0.25*(h[my_neighbour_left]+h[my_neighbour_right]+h[my_neighbour_top]+h[my_neighbour_bottom]);
}
