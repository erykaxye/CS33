//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name: Erica Xie
 * UCLA ID: 404920875
 * Email id: ericaxie@ucla.edu
 * Input: New files
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
	double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	int x, y, z, step;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, (double)(3 * stepCount));

	for(step = 0; step < stepCount; step++) 
	{

		#pragma omp parallel for private (y)
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y+=4)
			{
				u[Index(0, y, z)] *= boundryScale; 
				u[Index(0, y + 1, z)] *= boundryScale;
				u[Index(0, y + 2, z)] *= boundryScale;
				u[Index(0, y + 3, z)] *= boundryScale; 
			}
		}

		#pragma omp parallel for private (x, y) 
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for( x = 1; x < xMax; x++)
				{
					u[Index(x, y, z)] += u[Index(x - 1, y, z)] * nu;
			//		u[Index(x + 1, y, z)] += u[Index(x, y, z)] * nu;
				}
			}
		}

		#pragma omp parallel for private (y)
		for(z = 0; z < yMax; z++)
		{
			for(y = 0; y < yMax; y+=4)
			{
				u[Index(0, y, z)] *= boundryScale;
				u[Index(0, y + 1, z)] *= boundryScale;
				u[Index(0, y + 2, z)] *= boundryScale;
				u[Index(0, y + 3, z)] *= boundryScale;
			}
		}

		for(z = 0; z < zMax; z++)
		{
			#pragma omp parallel for private (x)
			for(y = 0; y < yMax; y++) 
			{
				for(x = xMax - 2; x >= 0; x--)
				{
					u[Index(x, y, z)] += u[Index(x + 1, y, z)] * nu;
				}
			}
		}

		#pragma omp parallel for private (x)
		for(z = 0; z < zMax; z++)
		{
			for(x = 0; x < xMax; x+=4)
			{
				u[Index(x, 0, z)] *= boundryScale;
				u[Index(x + 1, 0, z)] *= boundryScale;
				u[Index(x + 2, 0, z)] *= boundryScale;
				u[Index(x + 3, 0, z)] *= boundryScale;
			}
		}

		#pragma omp parallel for private (x, y) //changed 
		for(z = 0; z < zMax; z++)
		{
			for(y = 1; y < yMax; y++)
			{
				for(x = 0; x < xMax; x+=2)
				{
					u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
					u[Index(x + 1, y, z)] += u[Index(x + 1, y - 1, z)] * nu;
				}
			}
		}

		#pragma omp parallel for private (x)
		for(z = 0; z < zMax; z++)
		{
			for(x = 0; x < xMax; x+=4)
			{
				u[Index(x, yMax - 1, z)] *= boundryScale;
				u[Index(x + 1, yMax - 1, z)] *= boundryScale;
				u[Index(x + 2, yMax - 1, z)] *= boundryScale;
				u[Index(x + 3, yMax - 1, z)] *= boundryScale;
			}
		}

		#pragma omp parallel for private (x, y)
		for(z = 0; z < zMax; z++)
		{
			for(y = yMax - 2; y >= 0; y--)
			{
				for(x = 0; x < xMax; x+=2)
				{
					u[Index(x, y, z)] += u[Index(x, y + 1, z)] * nu;
					u[Index(x + 1, y, z)] += u[Index(x + 1, y + 1, z)] * nu;
				}
			}
		}

		#pragma omp parallel for private (x)
		for(y = 0; y < yMax; y++)
		{
			for(x = 0; x < xMax; x+=4)
			{
				u[Index(x, y, 0)] *= boundryScale;
				u[Index(x + 1, y, 0)] *= boundryScale;
				u[Index(x + 2, y, 0)] *= boundryScale;
				u[Index(x + 3, y, 0)] *= boundryScale;
			}
		}
 
		for(z = 1; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 0; x < xMax; x+=2)
				{
					u[Index(x, y, z)] = u[Index(x, y, z - 1)] * nu;
					u[Index(x + 1, y, z)] = u[Index(x + 1, y, z - 1)] * nu;
				}
			}
		}

		#pragma omp parallel for private (x)
		for(y = 0; y < yMax; y++)
		{
			for(x = 0; x < xMax; x+=4)
			{
				u[Index(x, y, zMax - 1)] *= boundryScale;
				u[Index(x + 1, y, zMax - 1)] *= boundryScale;
				u[Index(x + 2, y, zMax - 1)] *= boundryScale;
				u[Index(x + 3, y, zMax - 1)] *= boundryScale;
			}
		}

		for(z = zMax - 2; z >= 0; z--)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 0; x < xMax; x+=2)
				{
					u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
					u[Index(x + 1, y, z)] += u[Index(x + 1, y, z + 1)] * nu;
				}
			}
		}
	}
	#pragma omp parallel for private (x, y)
	for(z = 0; z < zMax; z++)
	{
		for(y = 0; y < yMax; y++)
		{
			for(x = 0; x < xMax; x+=4)
			{
				u[Index(x, y, z)] *= postScale;
				u[Index(x + 1, y, z)] *= postScale;
				u[Index(x + 2, y, z)] *= postScale;
				u[Index(x + 3, y, z)] *= postScale;
			}
		}
	}
}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
//		#pragma omp parallel for private (x, y)
		for(z = 1; z < zMax - 1; z++)
		{ 
#pragma omp parallel for private (x)
			for(y = 1; y < yMax - 1; y++)
			{ 
      				for(x = 1; x < xMax - 1; x+=2)
				{
int x1 = x + 1;
int x2 = x - 1;
double index1 = u[Index(x, y, z)];
double index2 = u[Index(x1, y, z)];

					g[Index(x, y, z)] = 1.0 / sqrt(epsilon + 
						SQR(index1 - index2) + 
						SQR(index1 - u[Index(x2, y, z)]) + 
						SQR(index1 - u[Index(x, y + 1, z)]) + 
						SQR(index1 - u[Index(x, y - 1, z)]) + 
						SQR(index1 - u[Index(x, y, z + 1)]) + 
						SQR(index1 - u[Index(x, y, z - 1)]));

 					g[Index(x1, y, z)] = 1.0 / sqrt(epsilon +
                                                SQR(index2 - u[Index(x + 2, y, z)]) +
                                                SQR(index2 - index1) +
                                                SQR(index2 - u[Index(x1, y + 1, z)]) +
                                                SQR(index2 - u[Index(x1, y - 1, z)]) +
                                                SQR(index2 - u[Index(x1, y, z + 1)]) +
                                                SQR(index2 - u[Index(x1, y, z - 1)]));

				}
			}
		}
		memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
		OMP_GaussianBlur(conv, Ksigma, 3);
		#pragma omp parallel for private (x, y)
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 0; x < xMax; x+=4)
				{
int index1 = Index(x, y, z);
int index2 = Index(x + 1, y, z);
int index3 = Index(x + 2, y, z);
int index4 = Index(x + 3, y, z);
					double r = conv[index1] * f[index1] / sigma2;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[index1] -= f[index1] * r;

double rl = conv[index2] * f[index2] / sigma2;
                                        rl = (rl * (2.38944 + rl * (0.950037 + rl))) / (4.65314 + rl * (2.57541 + rl * (1.48937 + rl)));
                                        conv[index2] -= f[index2] * rl;

double rm = conv[index3] * f[index3] / sigma2;
                                        rm = (rm * (2.38944 + rm * (0.950037 + rm))) / (4.65314 + rm * (2.57541 + rm * (1.48937 + rm)));
                                        conv[index3] -= f[index3] * rm;

double rn = conv[index4] * f[index4] / sigma2;
                                        rn = (rn * (2.38944 + rn * (0.950037 + rn))) / (4.65314 + rn * (2.57541 + rn * (1.48937 + rn)));
                                        conv[index4] -= f[index4] * rn;

				}
			}
		}
		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;
		for(z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for(x = 1; x < xMax - 1; x++)
				{
int x1 = x + 1;
int x2 = x - 1;
int index1 = Index(x, y, z);
int index2 = Index(x1, y, z);
int index3 = Index(x2, y, z);

					double oldVal = u[index1];
					double newVal = (u[index1] + dt * ( 
						u[index3] * g[index3] + 
						u[index2] * g[index2] + 
						u[Index(x, y - 1, z)] * g[Index(x, y - 1, z)] + 
						u[Index(x, y + 1, z)] * g[Index(x, y + 1, z)] + 
						u[Index(x, y, z - 1)] * g[Index(x, y, z - 1)] + 
						u[Index(x, y, z + 1)] * g[Index(x, y, z + 1)] - gamma * conv[index1])) /
						(1.0 + dt * (g[index2] + g[index3] + g[Index(x, y + 1, z)] + g[Index(x, y - 1, z)] + g[Index(x, y, z + 1)] + g[Index(x, y, z - 1)]));
					if(fabs(oldVal - newVal) < epsilon)
					{
						converged++;
					}
					u[index1] = newVal;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}

