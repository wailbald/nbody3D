//
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

//
typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;

//
typedef struct particle_s {

  f32 *restrict x, *restrict y, *restrict z;
  f32 *restrict vx, *restrict vy, *restrict vz;
  
} particle_t;

//
void init(particle_t *p, u64 n)
{
  for (u64 i = 0; i < n; i++)
    {
      //
      u64 r1 = (u64)rand();
      u64 r2 = (u64)rand();
      f32 sign = (r1 > r2) ? 1 : -1;
      
      //
      p->x[i] = sign * (f32)rand() / (f32)RAND_MAX;
      p->y[i] = (f32)rand() / (f32)RAND_MAX;
      p->z[i] = sign * (f32)rand() / (f32)RAND_MAX;

      //
      p->vx[i] = (f32)rand() / (f32)RAND_MAX;
      p->vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
      p->vz[i] = (f32)rand() / (f32)RAND_MAX;
    }
}

//
void move_particles(particle_t *p, f32 dt, u64 n)
{
  //
  const f32 softening = 1e-20;

  __m256 x1, y1, z1, x2, y2, z2, vx, vy, vz;
  __m256 dx, dy, dz, dx2, dy2, dz2, sof, tmp;
  __m256 fx, fy, fz, d_2, d_3_over_2, mdt;

  sof[0] = softening;
  sof[1] = softening;
  sof[2] = softening;
  sof[3] = softening;
  sof[4] = softening;
  sof[5] = softening;
  sof[6] = softening;
  sof[7] = softening;

  mdt[0] = dt;
  mdt[1] = dt;
  mdt[2] = dt;
  mdt[3] = dt;
  mdt[4] = dt;
  mdt[5] = dt;
  mdt[6] = dt;
  mdt[7] = dt;

  //
  for (u64 i = 0; i < n; i+=8)
  {
      //
      fx = _mm256_setzero_ps();
      fy = _mm256_setzero_ps();
      fz = _mm256_setzero_ps();

      x1 = _mm256_loadu_ps(p->x+i);
      y1 = _mm256_loadu_ps(p->y+i);
      z1 = _mm256_loadu_ps(p->z+i);
      
      vx = _mm256_loadu_ps(p->vx+i);
      vy = _mm256_loadu_ps(p->vy+i);
      vz = _mm256_loadu_ps(p->vz+i);

      //23 floating-point operations
    for (u64 j = 0; j < n; j++)
    {
      x2 = _mm256_loadu_ps(p->x+j);
      y2 = _mm256_loadu_ps(p->y+j);
      z2 = _mm256_loadu_ps(p->z+j);

      //Newton's law
      dx = _mm256_sub_ps(x2,x1);
      dy = _mm256_sub_ps(y2,y1);
      dz = _mm256_sub_ps(z2,z1);

      dx2 = _mm256_mul_ps(dx,dx);
      dy2 = _mm256_mul_ps(dy,dy);
      dz2 = _mm256_mul_ps(dz,dz);

      d_2 = _mm256_add_ps(dx2,dy2);
      d_2 = _mm256_add_ps(d_2,dz2);
      d_2 = _mm256_add_ps(d_2,sof);

      d_2 = _mm256_rsqrt_ps(d_2);

      d_3_over_2 = _mm256_mul_ps(d_2,d_2);
      d_3_over_2 = _mm256_mul_ps(d_3_over_2,d_2);

      fx = _mm256_fmadd_ps(dx,d_3_over_2,fx);
      fy = _mm256_fmadd_ps(dy,d_3_over_2,fy);
      fz = _mm256_fmadd_ps(dz,d_3_over_2,fz);
    }

    //
    vx = _mm256_fmadd_ps(mdt,fx,vx);
    vy = _mm256_fmadd_ps(mdt,fy,vy);
    vz = _mm256_fmadd_ps(mdt,fz,vz);

    _mm256_storeu_ps(p->vx+i,vx);
    _mm256_storeu_ps(p->vy+i,vy);
    _mm256_storeu_ps(p->vz+i,vz);
  }
  printf("vit = %lf\n",p->vx[0]);

  //3 floating-point operations
  for (u64 i = 0; i < n; i+=8)
    {
      x1 = _mm256_loadu_ps(p->x+i);
      y1 = _mm256_loadu_ps(p->y+i);
      z1 = _mm256_loadu_ps(p->z+i);

      vx = _mm256_loadu_ps(p->vx+i);
      vy = _mm256_loadu_ps(p->vy+i);
      vz = _mm256_loadu_ps(p->vz+i);

      x1 = _mm256_fmadd_ps(mdt,vx,x1);
      y1 = _mm256_fmadd_ps(mdt,vy,y1);
      z1 = _mm256_fmadd_ps(mdt,vz,z1);

      _mm256_storeu_ps(p->x+i,x1);
      _mm256_storeu_ps(p->y+i,y1);
      _mm256_storeu_ps(p->z+i,z1);
    }
    printf("val = %lf\n",p->x[0]); 
}

//
int main(int argc, char **argv)
{
  //
  const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
  const u64 steps= 10;
  f32 dt = 0.01;

  //
  f64 rate = 0.0, drate = 0.0;

  //Steps to skip for warm up
  const u64 warmup = 3;
  
  //
  particle_t *p = malloc(sizeof(particle_t) * n);

  p->x = aligned_alloc(64,sizeof(f32) * n);
  p->y = aligned_alloc(64,sizeof(f32) * n);
  p->z = aligned_alloc(64,sizeof(f32) * n);

  p->vx = aligned_alloc(64,sizeof(f32) * n);
  p->vy = aligned_alloc(64,sizeof(f32) * n);
  p->vz = aligned_alloc(64,sizeof(f32) * n);

  //
  init(p, n);

  const u64 s = sizeof(particle_t) * n;
  
  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
  
  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  
  //
  for (u64 i = 0; i < steps; i++)
    {
      //Measure

      const f64 start = omp_get_wtime();

      move_particles(p, dt, n);

      const f64 end = omp_get_wtime();

      //Number of interactions/iterations
      const f32 h1 = (f32)(n) * (f32)(n - 1);

      //GFLOPS
      const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;
      
      if (i >= warmup)
  {
    rate += h2 / (end - start);
    drate += (h2 * h2) / ((end - start) * (end - start));
  }

      //
      printf("%5llu %10.3e %10.3e %8.1f %s\n",
       i,
       (end - start),
       h1 / (end - start),
       h2 / (end - start),
       (i < warmup) ? "*" : "");
      
      fflush(stdout);
    }

  //
  rate /= (f64)(steps - warmup);
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
   "Average performance:", "", rate, drate);
  printf("-----------------------------------------------------\n");
  
  //
  free(p->x);
  free(p->y);
  free(p->z);

  free(p->vx);
  free(p->vy);
  free(p->vz);

  free(p);
  //
  return 0;
}
