/*******************************
vat
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

int VAT(char* dataFile,int N);

int main(int argc, char* argv[])
{
	printf("    vat    \n");
	int N=atoi(argv[2]);
	VAT(argv[1],N);
	return 0;
}

int VAT(char* dataFile,int N)
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
	float *distArray=NULL;
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
	printf("nodeNUm = %d \n",nodeNum);

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

	/**********************generate new order**********************/

	int *I=(int *)malloc(sizeof(int)*nodeNum);
	unsigned char*J=(unsigned char *)malloc(sizeof(unsigned char)*(nodeNum));
	float *lowcost=(float *)malloc(sizeof(float)*nodeNum);
	memset(I,0,sizeof(int)*nodeNum);
	//lowcost
	for(i=0;i<nodeNum;i++)
	{
		lowcost[i]=distArray[isel*nodeNum+i];
		J[i]=0;
	}
	J[isel] = 1;
	I[0]=isel;
	for(i=1;i<nodeNum;i++)
	{
		dmin=dmax;
		for(j=0;j<nodeNum;j++)
		{
			if(lowcost[j] < dmin&&J[j]==0)
			{
				isel=j;
				dmin=lowcost[j];
			}
		}
		J[isel]=1;
		I[i]=isel;
		for(j=0;j<nodeNum;j++)
		{
			if(J[j]==0&&lowcost[j]>distArray[isel*nodeNum+j])
			{
				lowcost[j]=distArray[isel*nodeNum+j];
			}
		}

	}

	//generate new distance array
	unsigned char tempC;
	unsigned char *distArrayN=(unsigned char *)malloc(sizeof(unsigned char)*nodeNum*nodeNum);
	for(i=0;i<nodeNum;i++)
	{
		for(j=0;j<i+1;j++)
		{
			tempC=(unsigned char)(distArray[I[i]*nodeNum+I[j]]/dmax*255);
			distArrayN[i*nodeNum+j]=tempC;
			distArrayN[j*nodeNum+i]=tempC;
		}
	}
	
	/*****************show the image*******************/
	//char name[20];
	p=strtok(dataFile,".");
	sprintf(name,"VAT_%s.jpg",p);
	IplImage *img=cvCreateImageHeader(cvSize(nodeNum,nodeNum),IPL_DEPTH_8U,1);
	IplImage *img1=cvCreateImage(cvSize(1000,1000),IPL_DEPTH_8U,1);
	cvSetData(img,distArrayN,nodeNum);

	cvResize(img,img1,CV_INTER_NN);  //resize image to 1000*1000

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

