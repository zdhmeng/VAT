/*******************************
efivat
******************************/ 

//#include <stdafx.h>   //windows
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#define LEN 10000
#define MAXLEN 20
using namespace cv; 
using namespace std;

int iVAT(char* dataFile,int N);
int iiVAT(char* dataFile,int N);

int main(int argc, char* argv[])
{
	printf("    efivat    \n");
	int N=atoi(argv[2]);
	iVAT(argv[1],N);
	return 0;
}

int iVAT(char* dataFile,int N)
{
	//load data
	FILE *fdata=fopen(dataFile,"r");
	//fail to load data
	if(!fdata)
	{
		printf("Fail to load the data\n");
		return 0;
	}
	//txt to array
	int nodeNum=0;
	int nodeArrayLen=LEN*N;
	int onelinelen=MAXLEN*N;
	const char *delim = ",";
	char *p;
	
	char *line=(char *)malloc(onelinelen*sizeof(char));
	char name[20];
	float *distArray=NULL,*distArrayC=NULL;
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
	distArray=(float *)malloc(nodeNum*nodeNum*sizeof(float));
	distArrayC=(float *)malloc(nodeNum*nodeNum*sizeof(float));
	nodeArrayLen=nodeNum*N;
	nodeArray=(float *)realloc(nodeArray,nodeArrayLen*sizeof(float));
	fclose(fdata);
	free(line);

	//original dist
	float sum;
	float dmax=0,dmin;
	float temp;
	int isel;

	if(distArray==NULL)
	{
		printf("CPU Failed to malloc\n");
		return 0;
	}
	int j,in,jn,k;
	for(i=0;i<nodeNum;i++)
	{
		in=i*N;
		for(j=0;j<nodeNum;j++)
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
	printf("nodeNum = %d \n",nodeNum);

	/**********************max**********************/
	for(i=0;i<nodeNum;i++)
	{
		in=i*nodeNum;
		for(j=0;j<i;j++)
		{
			if(distArray[in+j]>dmax)
			{
				dmax=distArray[in+j];
				isel=i;
			}
			
		}
	}
	printf("isel is %d\n",isel);
	
	/**********************generate new order**********************/
	int *I=(int *)malloc(sizeof(int)*nodeNum);
	bool*J=(bool *)malloc(sizeof(bool)*(nodeNum));
	float *lowcost=(float *)malloc(sizeof(float)*nodeNum);
	memset(I,0,sizeof(int)*nodeNum);
	//lowcost
	for(int i=0;i<nodeNum;i++)
	{
		lowcost[i]=distArray[isel*nodeNum+i];
		J[i]=0;
	}
	J[isel] = 1;
	I[0]=isel;
	for(int i=1;i<nodeNum;i++)
	{
		dmin=dmax;
		for(int j=0;j<nodeNum;j++)
		{
			if(lowcost[j] < dmin&&J[j]==0)
			{
				isel=j;
				dmin=lowcost[j];
			}
		}
		J[isel]=1;
		I[i]=isel;
		for(int j=0;j<nodeNum;j++)
		{
			if(J[j]==0&&lowcost[j]>distArray[isel*nodeNum+j])
			{
				lowcost[j]=distArray[isel*nodeNum+j];
			}
		}

	}

	//generate new distance array
	unsigned char *distArrayN=(unsigned char *)malloc(sizeof(unsigned char)*nodeNum*nodeNum);
	for(i=0;i<nodeNum;i++)
	{
		for(j=0;j<i+1;j++)
		{
			temp=distArray[I[i]*nodeNum+I[j]];
			distArrayC[i*nodeNum+j]=temp;
			distArrayC[j*nodeNum+i]=temp;
		}
	}

	/*****************generate image*******************/
	for(unsigned int i=1;i<nodeNum;i++)
	{
		temp=dmax+1;
		for(k=0;k<i;k++)
		{
			if(distArrayC[i*nodeNum+k]<temp)
			{
				temp=distArrayC[i*nodeNum+k];
				isel=k;
			}
		}
		distArrayC[i*nodeNum+isel]=temp;
		for(k=0;k<i;k++)
		{
			distArrayC[i*nodeNum+k]=temp;
			if(distArrayC[isel*nodeNum+k]>temp)
			{
				distArrayC[i*nodeNum+k]=distArrayC[isel*nodeNum+k];
			}
			distArrayC[k*nodeNum+i]=distArrayC[i*nodeNum+k];
			if(distArrayC[i*nodeNum+k]>dmin)
			{
				dmin=distArrayC[i*nodeNum+k];
			}
		}
	}
	for(unsigned int i=0;i<nodeNum*nodeNum;i++)
	{
		distArrayN[i]=(unsigned char)(distArrayC[i]/dmin*255);
	}
	
	/*****************show the image*******************/
	//char name[20];
	p=strtok(dataFile,".");
	sprintf(name,"efiVAT_%s.jpg",p);
	IplImage *img=cvCreateImageHeader(cvSize(nodeNum,nodeNum),IPL_DEPTH_8U,1);
	IplImage *img1=cvCreateImage(cvSize(1000,1000),IPL_DEPTH_8U,1);
	cvSetData(img,distArrayN,nodeNum);

	cvResize(img,img1,CV_INTER_NN);

	cvNamedWindow("img");
	cvSaveImage(name,img);
	cvShowImage("img",img1);
	cvWaitKey(0);
	cvDestroyWindow("img");
	cvReleaseImageHeader(&img);
	cvReleaseImage(&img1);

	free(I);
	free(J);
	free(lowcost);
	free(distArray);
	free(nodeArray);
	free(distArrayN);
	return 0;
}



