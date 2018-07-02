/***************************************
pmivat  version:01
2017-02-24
****************************************/
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LEN 10000
#define MAXLEN 20
#define THREADSP 1024	//Neworder blocks
#define THREADSC 256	//Neworder threads
#define THREADSM 256  //maxdist threads
#define THREADSD 256	//newDist threads

using namespace std;
using namespace cv;

/************************GPU**********************/
//maxdist   <<<1024,128|256>>> 1
__global__ void maxDist(int nodeNum,int nodeNum2, float *DdistArray ,float *min256,int *i256)
{
	__shared__ float maxCache[THREADSM];
	__shared__ int iselCache[THREADSM];
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int isel=tid;
	float max=DdistArray[tid];
	while (tid<nodeNum2)
	{
		if(DdistArray[tid]>max)
		{
			max=DdistArray[tid];
			isel=tid;
		}
		tid+=gridDim.x*blockDim.x;
	}
	maxCache[threadIdx.x]=max;
	iselCache[threadIdx.x]=isel;
	
	__syncthreads();
	
	if(threadIdx.x<128)
	{
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+128])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+128];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+128];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<64)
	{
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+64])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+64];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+64];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<32)
	{
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+32])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+32];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+32];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+16])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+16];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+16];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+8])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+8];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+8];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+4])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+4];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+4];
		}	
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+2])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+2];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+2];
		}		
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+1])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+1];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+1];
		}
	}
	if(threadIdx.x==0)
	{
		min256[blockIdx.x]=maxCache[0];
		i256[blockIdx.x]=iselCache[0]/nodeNum;
	}
}

//max<<<1,THREADSP>>> 2
__global__ void maxi(float *min256,int *i256,float *Dmax,char *DJ,int *Neworder)
{
	__shared__ float minCache[THREADSP];
	__shared__ int iselCache[THREADSP];
	
	minCache[threadIdx.x]=min256[threadIdx.x];
	iselCache[threadIdx.x]=i256[threadIdx.x];
	__syncthreads();
	
	if(threadIdx.x<512)
	{
		if(minCache[threadIdx.x]<minCache[threadIdx.x+512])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+512];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+512];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<256)
	{
		if(minCache[threadIdx.x]<minCache[threadIdx.x+256])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+256];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+256];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<128)
	{
		if(minCache[threadIdx.x]<minCache[threadIdx.x+128])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+128];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+128];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<64)
	{
		if(minCache[threadIdx.x]<minCache[threadIdx.x+64])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+64];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+64];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<32)
	{
		if(minCache[threadIdx.x]<minCache[threadIdx.x+32])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+32];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+32];
		}
		if(minCache[threadIdx.x]<minCache[threadIdx.x+16])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+16];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+16];
		}
		if(minCache[threadIdx.x]<minCache[threadIdx.x+8])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+8];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+8];
		}
		if(minCache[threadIdx.x]<minCache[threadIdx.x+4])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+4];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+4];
		}	
		if(minCache[threadIdx.x]<minCache[threadIdx.x+2])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+2];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+2];
		}		
		if(minCache[threadIdx.x]<minCache[threadIdx.x+1])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+1];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+1];
		}
	}
	if(threadIdx.x==0)
	{
		DJ[iselCache[0]]=1;
		Neworder[0]=iselCache[0];
		*Dmax=minCache[0];
	}
}

/**********initial lowCost 3**********/
__global__ void initial_lowCost(float *DdistArray,float *lowCost,float dmax,int nodeNum,float *lowArray)
{
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	if(tid<nodeNum)
	{
		lowCost[tid]=dmax+1;
		lowArray[tid]=0;
	}
}
/**********************Neworder************************/

//find the mininum of lowCost 5
__global__ void minlowCost(float *DdistArray,float *lowCost,float *min256,int *i256,float dmax,int nodeNum,int *Neworder,char *DJ,int i,int *Dlength,int length,int miniTHREAD,float *lowArray)
{
	__shared__ float minCache[THREADSC];
	__shared__ int iselCache[THREADSC];
	
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int index_i=Neworder[i-1];
	int index=index_i*nodeNum+tid;
	int right=0;

	if(tid<nodeNum)
	{
		iselCache[threadIdx.x]=tid;
		right=DdistArray[index]<lowCost[tid];
		minCache[threadIdx.x]=right*DdistArray[index]+(!right)*lowCost[tid]+DJ[tid]*(dmax+1);
		lowCost[tid]=minCache[threadIdx.x];
	}
	else
	{
		minCache[threadIdx.x]=dmax+1;
		
	}
	__syncthreads();

	if(threadIdx.x<128)
	{
		if(minCache[threadIdx.x]>minCache[threadIdx.x+128])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+128];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+128];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<64)
	{
		if(minCache[threadIdx.x]>minCache[threadIdx.x+64])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+64];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+64];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<32)
	{
		if(minCache[threadIdx.x]>minCache[threadIdx.x+32])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+32];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+32];
		}
		if(minCache[threadIdx.x]>minCache[threadIdx.x+16])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+16];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+16];
		}
		if(minCache[threadIdx.x]>minCache[threadIdx.x+8])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+8];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+8];
		}
		if(minCache[threadIdx.x]>minCache[threadIdx.x+4])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+4];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+4];
		}	
		if(minCache[threadIdx.x]>minCache[threadIdx.x+2])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+2];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+2];
		}		
		if(minCache[threadIdx.x]>minCache[threadIdx.x+1])
		{
			minCache[threadIdx.x]=minCache[threadIdx.x+1];
			iselCache[threadIdx.x]=iselCache[threadIdx.x+1];
		}
	}
	
	if(threadIdx.x==0)
	{
		min256[blockIdx.x]=minCache[0];
		i256[blockIdx.x]=iselCache[0];
		iselCache[0]=atomicAdd(Dlength,1);
		//printf("From GPU Dlength = %d\n",right+1);
	}
	__syncthreads();
	
	if(iselCache[0]>=length-1)
	{
		minCache[threadIdx.x]=dmax+1;
		if(threadIdx.x<length)
		{
			minCache[threadIdx.x]=min256[threadIdx.x];
			iselCache[threadIdx.x]=i256[threadIdx.x];
		}
		__syncthreads();
		
		if(miniTHREAD>=256)
		{
			if(threadIdx.x<128)
			{
				if(minCache[threadIdx.x]>minCache[threadIdx.x+128])
				{
					minCache[threadIdx.x]=minCache[threadIdx.x+128];
					iselCache[threadIdx.x]=iselCache[threadIdx.x+128];
				}
			}
			__syncthreads();
		}
		if(miniTHREAD>=128)
		{
			if(threadIdx.x<64)
			{
				if(minCache[threadIdx.x]>minCache[threadIdx.x+64])
				{
					minCache[threadIdx.x]=minCache[threadIdx.x+64];
					iselCache[threadIdx.x]=iselCache[threadIdx.x+64];
				}
			}
			__syncthreads();
		}
		if(threadIdx.x<32)
		{
			if(minCache[threadIdx.x]>minCache[threadIdx.x+32])
			{
				minCache[threadIdx.x]=minCache[threadIdx.x+32];
				iselCache[threadIdx.x]=iselCache[threadIdx.x+32];
			}
			if(minCache[threadIdx.x]>minCache[threadIdx.x+16])
			{
				minCache[threadIdx.x]=minCache[threadIdx.x+16];
				iselCache[threadIdx.x]=iselCache[threadIdx.x+16];
			}
			if(minCache[threadIdx.x]>minCache[threadIdx.x+8])
			{
				minCache[threadIdx.x]=minCache[threadIdx.x+8];
				iselCache[threadIdx.x]=iselCache[threadIdx.x+8];
			}
			if(minCache[threadIdx.x]>minCache[threadIdx.x+4])
			{
				minCache[threadIdx.x]=minCache[threadIdx.x+4];
				iselCache[threadIdx.x]=iselCache[threadIdx.x+4];
			}	
			if(minCache[threadIdx.x]>minCache[threadIdx.x+2])
			{
				minCache[threadIdx.x]=minCache[threadIdx.x+2];
				iselCache[threadIdx.x]=iselCache[threadIdx.x+2];
			}		
			if(minCache[threadIdx.x]>minCache[threadIdx.x+1])
			{
				minCache[threadIdx.x]=minCache[threadIdx.x+1];
				iselCache[threadIdx.x]=iselCache[threadIdx.x+1];
			}
		}
		if(threadIdx.x==0)
		{
			DJ[iselCache[0]]=1;
			Neworder[i]=iselCache[0];
			lowArray[i-1]=minCache[0];
			//printf("From GPU %d\n",*Dlength);
			*Dlength=0;
		}
		
	}
	
}
//max of lowArray
__global__ void maxlowArray(float *lowArray,float *Dmax,int nodeNum,float *min256,int length,int *Dlength,int miniTHREAD)
{
	__shared__ float maxCache[THREADSC];
	__shared__ int last;
	int tid=tid=blockIdx.x*blockDim.x+threadIdx.x;
	
	if(tid<nodeNum-1)
	{
		maxCache[threadIdx.x]=lowArray[tid];
	}
	else
	{
		maxCache[threadIdx.x]=0;
	}
	__syncthreads();
	
	if(threadIdx.x<128)
	{
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+128])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+128];
		}
	}
	__syncthreads();
	if(threadIdx.x<64)
	{
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+64])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+64];
		}
	}
	__syncthreads();
	
	if(threadIdx.x<32)
	{
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+32])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+32];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+16])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+16];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+8])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+8];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+4])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+4];
		}
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+2])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+2];
		}		
		if(maxCache[threadIdx.x]<maxCache[threadIdx.x+1])
		{
			maxCache[threadIdx.x]=maxCache[threadIdx.x+1];
		}
	}
	
	
	if(threadIdx.x==0)
	{
		min256[blockIdx.x]=maxCache[0];
		last=atomicAdd(Dlength,1);	
	}
	__syncthreads();
	
	if(last>=length-1)
	{
		if(threadIdx.x<length)
		{
			maxCache[threadIdx.x]=min256[threadIdx.x];
		}
		__syncthreads();
		
		if(miniTHREAD>=256)
		{
			if(threadIdx.x<128)
			{
				if(maxCache[threadIdx.x]<maxCache[threadIdx.x+128])
				{
					maxCache[threadIdx.x]=maxCache[threadIdx.x+128];
				}
			}
			__syncthreads();
		}
		if(miniTHREAD>=128)
		{
			if(threadIdx.x<64)
			{
				if(maxCache[threadIdx.x]<maxCache[threadIdx.x+64])
				{
					maxCache[threadIdx.x]=maxCache[threadIdx.x+64];
				}
			}
			__syncthreads();
		}
		
		if(threadIdx.x<32)
		{
			if(maxCache[threadIdx.x]<maxCache[threadIdx.x+32])
			{
				maxCache[threadIdx.x]=maxCache[threadIdx.x+32];
			}
			if(maxCache[threadIdx.x]<maxCache[threadIdx.x+16])
			{
				maxCache[threadIdx.x]=maxCache[threadIdx.x+16];
			}
			if(maxCache[threadIdx.x]<maxCache[threadIdx.x+8])
			{
				maxCache[threadIdx.x]=maxCache[threadIdx.x+8];
			}
			if(maxCache[threadIdx.x]<maxCache[threadIdx.x+4])
			{
				maxCache[threadIdx.x]=maxCache[threadIdx.x+4];
			}
			if(maxCache[threadIdx.x]<maxCache[threadIdx.x+2])
			{
				maxCache[threadIdx.x]=maxCache[threadIdx.x+2];
			}		
			if(maxCache[threadIdx.x]<maxCache[threadIdx.x+1])
			{
				maxCache[threadIdx.x]=maxCache[threadIdx.x+1];
			}
		}
		
		if(threadIdx.x==0)
		{
			*Dmax=maxCache[0];
		}
	}
}
//**newDist 7
__global__ void newDist(int nodeNum,float *lowArray,unsigned char *DdistArrayN,float dmax)
{
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int i=blockIdx.x*blockDim.x;
	unsigned char tempN;
	float temp=lowArray[tid];
	if(tid<nodeNum)
	{
		DdistArrayN[tid*nodeNum+tid]=0;
		while(i<nodeNum)
		{
			if(i>tid)
			{
				tempN=(unsigned char)(temp/dmax*255);
				DdistArrayN[i*nodeNum+tid]=tempN;
				DdistArrayN[tid*nodeNum+i]=tempN;
				if(lowArray[i]>temp)
				{
					temp=lowArray[i];
				}
			}
			i+=1;
		}
	}
}

int pmivat(char* dataFile,int N)
{
	printf("gpu-eVAT\n");
    clock_t begin,end,end1,end2,end3;

	/********************load data********************/
	FILE *fdata=fopen(dataFile,"r");
	cudaSetDevice(2);

	if(!fdata)
	{
		printf("Fail to load the data\n");
		return 0;
	}
	int nodeNum=0,nodeNum2=0,length=0;
	int nodeArrayLen=LEN*N;
	int onelinelen=MAXLEN*N;
	const char *delim = ",";
	char *p;
	char *line=(char *)malloc(onelinelen*sizeof(char));
	float *nodeArray=(float *)malloc(nodeArrayLen*sizeof(float));

	int i=0;
	while(fgets(line,onelinelen,fdata)!=NULL)
	{
		nodeNum++;
		p=strtok(line, delim);
		while(p)
		{
			nodeArray[i]=atof(p);
			p = strtok(NULL, delim);
			i++;
			if(i>nodeArrayLen-1)
			{
				nodeArrayLen <<= 1;
				nodeArray=(float *)realloc(nodeArray,nodeArrayLen*sizeof(float));
			}
		}
	}
	nodeNum2=nodeNum*nodeNum;
	free(line);
	nodeArrayLen=nodeNum*N;
	nodeArray=(float *)realloc(nodeArray,nodeArrayLen*sizeof(float));
	printf("nodeNum = %d\n",nodeNum);
	printf("nodeNum*N = %d\n",i);
	printf("load data sucess\n");
	/****************distArray*************/
	float sum;
	float temp,dmax=0;
	float *distArray=NULL;
	cudaMallocHost((void **)&distArray,nodeNum2*sizeof(float));
	unsigned char *distArrayN=NULL;
	cudaMallocHost((void **)&distArrayN,sizeof(unsigned char)*nodeNum2);
	if(distArray==NULL)
	{
		printf("Failed to malloc\n");
		return 0;
	}
	int j,in,jn,k;
	for(i=0;i<nodeNum;i++)
	{
		in=i*N;
		for(j=0;j<i+1;j++)
		{
			jn=j*N;
			sum=0;
			for(k=0;k<N;k++)
			{
				sum+=pow((nodeArray[in+k]-nodeArray[jn+k]),2);
			}
			temp=sqrt(sum);
			distArray[i*nodeNum+j]=temp;
			distArray[j*nodeNum+i]=temp;
		}

	}
	free(nodeArray);
	printf("DdistArray Generated\n");

	
	char *DJ=NULL;
	int *Neworder=NULL,*i256=NULL,*Dlength;
	
	float *DdistArray=NULL,*Dmax=NULL,*min256=NULL,*lowCost=NULL,*lowArray=NULL;
	cudaMalloc((void **)&i256,sizeof(int)*THREADSP);
	cudaMalloc((void **)&min256,sizeof(float)*THREADSP);
	cudaMalloc((void **)&DJ,nodeNum*sizeof(char));
	cudaMalloc((void **)&lowCost,nodeNum*sizeof(float));
	cudaMalloc((void **)&lowArray,nodeNum*sizeof(float));
	cudaMalloc((void **)&Neworder,nodeNum*sizeof(int));
	cudaMalloc((void **)&Dmax,sizeof(float));
	cudaMalloc((void **)&Dlength,sizeof(int));
	cudaMalloc((void **)&DdistArray,nodeNum2*sizeof(float));
	unsigned char *DdistArrayN=(unsigned char *)DdistArray;
	
	if(Neworder==NULL||i256==NULL||min256==NULL||DJ==NULL||Dmax==NULL||DdistArray==NULL||lowCost==NULL||DdistArrayN==NULL)
	{
		printf("Failed to CudaMalloc\n");
		return 0;
	}
	cudaMemset(DJ,0,nodeNum*sizeof(char));

	/*******************VAT******************/
	//max
	begin=clock();
	cudaMemcpy(DdistArray,distArray,nodeNum2*sizeof(float),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	maxDist<<<THREADSP,THREADSM>>>(nodeNum,nodeNum2, DdistArray,min256,i256);
	maxi<<<1,THREADSP>>>(min256,i256,Dmax,DJ,Neworder);
	cudaDeviceSynchronize();
	cudaMemcpy(&dmax,Dmax,sizeof(float),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Newmax is %f\n",dmax);
	//Neworder
	length=nodeNum/THREADSC+1;
	int miniTHREAD=(int)pow(2.0,(int)(log(length)/log(2))+1);
	initial_lowCost<<<nodeNum/256+1,256>>>(DdistArray,lowCost,dmax,nodeNum,lowArray);
	cudaDeviceSynchronize();
	for(int i=1;i<nodeNum;i++)
	{
		minlowCost<<<nodeNum/THREADSC+1,THREADSC>>>(DdistArray,lowCost,min256,i256,dmax,nodeNum,Neworder,DJ,i,Dlength,length,miniTHREAD,lowArray);
	}
	cudaDeviceSynchronize();
	end1=clock();
	//new max
	maxlowArray<<<nodeNum/THREADSC+1,THREADSC>>>(lowArray,Dmax,nodeNum,min256,length,Dlength,miniTHREAD);
	cudaDeviceSynchronize();
	cudaMemcpy(&dmax,Dmax,sizeof(float),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	end2=clock();
	printf("Newmax takes %fs\n",(end2-end1)/1000.0);
	printf("Newmax is %f\n",dmax);
	//New Distarray
	newDist<<<nodeNum/512+1,512>>>(nodeNum,lowArray,DdistArrayN,dmax);
	cudaDeviceSynchronize();
	end3=clock();
	printf("NewDist takes %fs\n",(end3-end2)/1000.0);
	cudaMemcpy(distArrayN,DdistArrayN,sizeof(unsigned char)*nodeNum2,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	end=clock();
	printf("All takes %fs\n",(end-begin)/1000.0);

	
	//*/
	//time
	char name[20];
	p=strtok(dataFile,".");
	sprintf(name,"pmivat_%s.jpg",p);
	
	IplImage *img=cvCreateImageHeader(cvSize(nodeNum,nodeNum),IPL_DEPTH_8U,1);
	IplImage *img1=cvCreateImage(cvSize(500,500),IPL_DEPTH_8U,1);
	cvSetData(img,distArrayN,nodeNum);
	cvSaveImage(name,img);

	cvResize(img,img1,CV_INTER_NN);
	cvNamedWindow("img1");
	cvShowImage("img1",img1);

	cvWaitKey(0);
	cvDestroyWindow("img1");
	cvReleaseImageHeader(&img);
	cvReleaseImage(&img1);
	//*/
	cudaFree(DJ);
	cudaFree(i256);
	cudaFree(min256);
	cudaFree(Dmax);
	cudaFree(lowCost);
	cudaFree(lowArray);
	cudaFree(Neworder);
	cudaFree(DdistArray);
	cudaFreeHost(distArray);
	cudaFreeHost(distArrayN);
	
	return 0;
}

int main(int argc, char* argv[])
{
	printf("    pmivat    \n");
	int N=atoi(argv[2]);
	pmivat(argv[1],N);
	return 0;
}

