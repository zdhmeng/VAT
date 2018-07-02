/***************************************
vat  version:16
2017-01-14-02
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

using namespace cv;
using namespace std;

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

/**********initial lowArray 3**********/
__global__ void initial_lowArray(float *DdistArray,float *lowArray,float dmax,int nodeNum)
{
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	if(tid<nodeNum)
	{
		lowArray[tid]=dmax+1;
	}
}
/**********************Neworder************************/

//find the mininum of lowArray 5
__global__ void minlowArray(float *DdistArray,float *lowArray,float *min256,int *i256,float dmax,int nodeNum,int *Neworder,char *DJ,int i,int *Dlength,int length,int miniTHREAD)
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
		right=DdistArray[index]<lowArray[tid];
		minCache[threadIdx.x]=right*DdistArray[index]+(!right)*lowArray[tid]+DJ[tid]*(dmax+1);
		lowArray[tid]=minCache[threadIdx.x];
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
			//printf("From GPU %d\n",*Dlength);
			*Dlength=0;
		}
		
	}
	
}

//**newDist 7
__global__ void newDist(int nodeNum,int nodeNum2,float *DdistArray,unsigned char *DdistArrayN,int * Neworder,float dmax)
{
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int i=0,j=0,tid2=0;
	while(tid<nodeNum2)
	{
		i=Neworder[tid/nodeNum];
		j=Neworder[tid%nodeNum];
		tid2=i*nodeNum+j;
		DdistArrayN[tid]=(unsigned char)(DdistArray[tid2]/dmax*255);
		tid+=gridDim.x*blockDim.x;
	}
}

int pvat(char* dataFile,int N)
{
	printf("No.01-20\n");
    	clock_t begin,end,clock1,clock2;

	/********************load data********************/
	FILE *fdata=fopen(dataFile,"r");
	cudaSetDevice(1);

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
	unsigned char *DdistArrayN=NULL;
	float *DdistArray=NULL,*Dmax=NULL,*min256=NULL,*lowArray=NULL;
	cudaMalloc((void **)&i256,sizeof(int)*THREADSP);
	cudaMalloc((void **)&min256,sizeof(float)*THREADSP);
	cudaMalloc((void **)&DJ,nodeNum*sizeof(char));
	cudaMalloc((void **)&lowArray,nodeNum*sizeof(float));
	cudaMalloc((void **)&Neworder,nodeNum*sizeof(int));
	cudaMalloc((void **)&Dmax,sizeof(float));
	cudaMalloc((void **)&Dlength,sizeof(int));
	cudaMalloc((void **)&DdistArray,nodeNum2*sizeof(float));
	cudaMalloc((void **)&DdistArrayN,nodeNum2*sizeof(unsigned char));
	
	if(Neworder==NULL||i256==NULL||min256==NULL||DJ==NULL||Dmax==NULL||DdistArray==NULL||lowArray==NULL||DdistArrayN==NULL)
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
	//Neworder
	clock1=clock();
	length=nodeNum/THREADSC+1;
	int miniTHREAD=(int)pow(2.0,(int)(log(length)/log(2))+1);
	initial_lowArray<<<nodeNum/256+1,256>>>(DdistArray,lowArray,dmax,nodeNum);
	cudaDeviceSynchronize();
	for(int i=1;i<nodeNum;i++)
	{
		minlowArray<<<nodeNum/THREADSC+1,THREADSC>>>(DdistArray,lowArray,min256,i256,dmax,nodeNum,Neworder,DJ,i,Dlength,length,miniTHREAD);
	}
	cudaDeviceSynchronize();
	clock2=clock();
	printf("Neworder takes %f s\n",(clock2-clock1)/1000.0);
	/*/test
	int *neworder=(int *)malloc(sizeof(int)*nodeNum);
	bool *J=(bool *)malloc(sizeof(bool)*nodeNum);
	cudaMemcpy(neworder,Neworder,sizeof(int)*nodeNum,cudaMemcpyDeviceToHost);
	for(int i=0;i<nodeNum;i++)
	{
		J[i]=0;
	}
	for(int i=0;i<nodeNum;i++)
	{
		J[neworder[i]]=1;
	}
	for(int i=0;i<nodeNum;i++)
	{
		if(J[i]==0) 
		{
			printf(" failed  %d \n",i);
			break;
		}
	}
	free(neworder);
	/*/
	//New Distarray
	int Blocks=(int)(nodeNum2/1024)+1;
	printf("Blocks= %d\n",Blocks);
	newDist<<<65535,1024>>>(nodeNum,nodeNum2,DdistArray,DdistArrayN,Neworder,dmax);
	cudaDeviceSynchronize();
	cudaMemcpy(distArrayN,DdistArrayN,sizeof(unsigned char)*nodeNum2,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	end=clock();
	printf("All takes %fs\n",(end-begin)/1000.0);

	
	//*/
	//time
	char name[20];
	p=strtok(dataFile,".");
	sprintf(name,"Vat_%s.jpg",p);
	
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
	cudaFree(lowArray);
	cudaFree(Neworder);
	cudaFree(DdistArray);
	cudaFree(DdistArrayN);
	cudaFreeHost(distArray);
	cudaFreeHost(distArrayN);
	
	return 0;
}

int main(int argc, char* argv[])
{
	printf("    pvat    \n");
	int N=atoi(argv[2]);
	pvat(argv[1],N);
	return 0;
}

