#include <stdio.h>

__global__ void helloGPU(void)
{
printf(" From thread %d :  Sugan Nalla - GPU ! \n ", threadIdx.x);
}

int main(void)
{
 // From CPU
printf(" Sugan Nalla - CPU ! \n ");

helloGPU <<< 1, 10 >>>();
cudaDeviceReset();
return 0;
}
