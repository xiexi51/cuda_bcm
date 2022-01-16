#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "cufft.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <cassert>
#include <vector>
#include <algorithm>
#include <functional>

#define ROW 64
#define COL 64
#define BSIZE 96

using namespace std;

template <typename T1, typename T2>
void gen_bcm(T1** pbcm_full, T2** pbcm_and_x, int row, int col, int bsize) {
	cudaMallocHost(pbcm_full, sizeof(T1) * row * col * bsize * bsize);
	cudaMallocHost(pbcm_and_x, sizeof(T2) * (row + 1) * col * bsize);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < bsize; k++) {
				float ra = (rand() * 10.0 / (float)RAND_MAX - 5.0);
                (*pbcm_and_x)[i * bsize * col + j * bsize + k] = ra;
					for (int l = 0; l < bsize; l++) {
						int w = i * bsize * bsize * col + j * bsize + k * bsize * col + l * (bsize * col + 1);
						if (l >= bsize - k) {
							w -= bsize * bsize * col;
						}
						(*pbcm_full)[w] = ra;
					}
			}
		}
	}
    for (int k = 0; k < col * bsize; k++) {
        (*pbcm_and_x)[row * col * bsize + k] = (T2)(rand() * 10.0 / (float)RAND_MAX - 5.0);
    }
}

__global__ void bcm_dot_scale(cufftComplex* bcm, cufftComplex* x, int row, int col, int bsize){
    register int bcnt = (bsize >> 1) + 1;
    register float scale = 1.0 / bsize;
    register int t_base = threadIdx.x * bcnt;
    register int b_t_base = blockIdx.x * col * bcnt + t_base;
    register cufftComplex tmp_bcm, tmp_x;
#pragma unroll
    for (int i = 0; i < bcnt; i++) {
        tmp_bcm = bcm[b_t_base + i];
        tmp_x = x[t_base + i];
        bcm[b_t_base + i].x = (tmp_bcm.x * tmp_x.x - tmp_bcm.y * tmp_x.y) * scale;
        bcm[b_t_base + i].y = (tmp_bcm.x * tmp_x.y + tmp_bcm.y * tmp_x.x) * scale;
    }
}

__global__ void collect_result(float* bcm, float* re_bcm, int row, int col, int bsize) {
    register float sum = 0;
    register int b_t_base = blockIdx.x * col * bsize + threadIdx.x;
#pragma unroll
    for (int i = 0; i < col; i++) {
        sum += bcm[b_t_base + i * bsize];
    }
    re_bcm[blockIdx.x * bsize + threadIdx.x] = sum;
}

#define SHOW
#if !(ROW <=4 && COL <= 4 && BSIZE <=4)
#undef SHOW
#endif

int main()
{
    assert(sizeof(cufftReal) == sizeof(float));
    assert(sizeof(cufftComplex) == sizeof(float) * 2);
    assert(!(BSIZE & 1));

    srand(123);
    cout << setprecision(3);
    float *h_bcm_full, *h_re_matmul, time_ms;
    cufftReal *h_bcm_and_x, *h_re_bcm;

    cout << "generating bcm..." << endl;
    gen_bcm(&h_bcm_full, &h_bcm_and_x, ROW, COL, BSIZE);
    cudaMallocHost(&h_re_matmul, sizeof(float) * ROW * BSIZE);
    cudaMallocHost(&h_re_bcm, sizeof(cufftReal) * ROW * BSIZE);

#ifdef SHOW
    for (int i = 0; i < 2 * BSIZE; i++) {
        for (int j = 0; j < 2 * BSIZE; j++) {
            cout << h_bcm_full[i * COL * BSIZE + j] << '\t';
        }
        cout << '\n';
    }
    cout << '\n';
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2 * BSIZE; j++) {
            cout << h_bcm_and_x[i * COL * BSIZE + j] << '\t';
        }
        cout << '\n';
    }
    cout << "\n\n";
    for (int i = 0; i < 2 * BSIZE; i++) {
        cout << h_bcm_and_x[ROW * COL * BSIZE + i] << '\t';
    }
    cout << '\n' << endl;
#endif

    cout << "row: " << ROW << " col: " << COL << " block size: " << BSIZE << endl;

    float *bcm_full, *re_matmul, *re_bcm;
    cufftReal* bcm_and_x;    
    cudaMalloc(&bcm_full, sizeof(float) * ROW * COL * BSIZE * BSIZE);
    cudaMalloc(&re_matmul, sizeof(float) * ROW * BSIZE);
    cudaMalloc(&bcm_and_x, sizeof(cufftReal) * (ROW + 1) * COL * (BSIZE + 2));
    cudaMalloc(&re_bcm, sizeof(float) * ROW * BSIZE);
    cudaMemcpy(bcm_full, h_bcm_full, sizeof(float) * ROW * COL * BSIZE * BSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(bcm_and_x, h_bcm_and_x, sizeof(cufftReal) * (ROW + 1) * COL * BSIZE, cudaMemcpyHostToDevice);
    cudaEvent_t time1, time2, time3, time4, time5;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventCreate(&time3);
    cudaEventCreate(&time4);
    cudaEventCreate(&time5);

    cublasHandle_t handleb;
    cublasCreate(&handleb);

    cout << "cublas start..." << endl;
    cudaEventRecord(time1);
    const float _alpha = 1, _beta = 0;
    cublasSgemm(handleb, CUBLAS_OP_N, CUBLAS_OP_N, 1, ROW * BSIZE, COL * BSIZE, &_alpha, bcm_and_x + ROW * COL * BSIZE, 1, bcm_full, COL * BSIZE, &_beta, re_matmul, 1);
    cudaMemcpy(h_re_matmul, re_matmul, sizeof(float) * ROW * BSIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(time2);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&time_ms, time1, time2);
    cout << "cublas finish, " << time_ms << "ms" << endl;
    cublasDestroy(handleb);

    cudaFree(bcm_full);
    cudaFree(re_matmul);

    cufftHandle handlef1, handlef2;
    {
        int rank = 1;                           // --- 1D FFTs
        int n[] = { BSIZE };                 // --- Size of the Fourier transform
        int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
        int idist = BSIZE, odist = (BSIZE / 2 + 1); // --- Distance between batches
        int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
        int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
        int batch = (ROW + 1) * COL;                      // --- Number of batched executions
        cufftPlanMany(&handlef1, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
    }
    {
        int rank = 1;                           // --- 1D FFTs
        int n[] = { BSIZE };                 // --- Size of the Fourier transform
        int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
        int idist = BSIZE / 2 + 1, odist = BSIZE; // --- Distance between batches
        int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
        int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
        int batch = ROW * COL;                      // --- Number of batched executions
        cufftPlanMany(&handlef2, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, batch);
    }
    cout << "bcm start..." << endl;
    cudaEventRecord(time1);
    cufftExecR2C(handlef1, bcm_and_x, (cufftComplex*)bcm_and_x);
    
    /*float *temp = (float*)malloc(sizeof(float) * 20);
    cudaMemcpy(temp, bcm_and_x, sizeof(float) * 20, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; i++) {
        cout << temp[i] << '\t';
    }
    cout << '\n' << endl;*/

    cudaEventRecord(time2);
    bcm_dot_scale <<<ROW, COL>>> ((cufftComplex*)bcm_and_x, (cufftComplex*)(bcm_and_x + ROW * COL * (BSIZE + 2)), ROW, COL, BSIZE);
    cudaEventRecord(time3);
    cufftExecC2R(handlef2, (cufftComplex*)bcm_and_x, bcm_and_x);
    cudaEventRecord(time4);
    collect_result <<<ROW, BSIZE>>> (bcm_and_x, re_bcm, ROW, COL, BSIZE);
    cudaMemcpy(h_re_bcm, re_bcm, sizeof(float) * ROW * BSIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(time5);
    cudaEventSynchronize(time5);

    cudaEventElapsedTime(&time_ms, time1, time2);
    cout << "rfft finish, " << time_ms << "ms" << endl;
    cudaEventElapsedTime(&time_ms, time2, time3);
    cout << "element-wise dot and scale finish, " << time_ms << "ms" << endl;
    cudaEventElapsedTime(&time_ms, time3, time4);
    cout << "irfft finish, " << time_ms << "ms" << endl;
    cudaEventElapsedTime(&time_ms, time4, time5);
    cout << "bcm finish, " << time_ms << "ms" << endl;
    
    cufftDestroy(handlef1);
    cufftDestroy(handlef2);
    
#ifdef SHOW    
    float* temp = (float*)malloc(sizeof(float) * (ROW + 1) * COL * (BSIZE + 2));
    cudaMemcpy(temp, bcm_and_x, sizeof(float) * (ROW + 1) * COL * (BSIZE + 2), cudaMemcpyDeviceToHost);
    for (int i = 0; i < (ROW + 1) * COL * (BSIZE + 2); i++) {
        cout << temp[i] << '\t';
    }
    cout << '\n' << endl;
    free(temp);
#endif

    cudaFree(bcm_and_x);
    cudaFree(re_bcm);

#ifdef SHOW
    for (int i = 0; i < ROW * BSIZE; i++) {
        cout << h_re_matmul[i] << '\t';
    }
    cout << '\n' << endl;
    for (int i = 0; i < ROW * BSIZE; i++) {
        cout << h_re_bcm[i] << '\t';
    }
    cout << '\n' << endl;
#endif

	vector<float> v_blas{ h_re_matmul, h_re_matmul + ROW * BSIZE };
    vector<float> v_bcm{ h_re_bcm, h_re_bcm + ROW * BSIZE };
    transform(v_blas.begin(), v_blas.end(), v_bcm.begin(), v_blas.begin(), minus<float>());
    float max_abs_err = abs(*max_element(v_blas.begin(), v_blas.end(), [](const float& a, const float& b){
            return abs(a) < abs(b);
        }));

    cout << "max abs error = " << max_abs_err << endl;

    cudaFree(h_bcm_full);
    cudaFree(h_bcm_and_x);
    cudaFree(h_re_matmul);
    cudaFree(h_re_bcm);
    return 0;
}