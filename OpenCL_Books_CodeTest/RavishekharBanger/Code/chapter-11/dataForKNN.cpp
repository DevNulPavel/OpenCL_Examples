// dataForKNN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include  <cstdlib>
#include <iostream>
#include <fstream>

#define DATA_SIZE 1024
#define NUM_CLASSES 3
using namespace std;

bool readkNNData(int*pX1, int* pX2, int *pY, int dataSize);

int _tmain(int argc, _TCHAR* argv[])
{
	ofstream myfile;
	myfile.open ("kNNData.txt");
	int class_selector;
	int x1, x2, y;
	for(int i=0; i< DATA_SIZE;++i)
	{
		class_selector = rand()%NUM_CLASSES;
		if(0 == class_selector)
		{
			x1 = rand()%10 + 100;
			x2 = rand()%5 + 1;
            y = 0;
		}
		else if (1 == class_selector)
		{
			x1 = rand()%5 + 5;
			x2 = rand()%5 + 50;
            y = 1;
		}
		else
		{
			x1 = rand()%5 + 5;
			x2 = rand()%5 + 1;
            y = 2;

		}

		//myfile <<x1<<","<<x2<<","<<y<< "\n";
		myfile <<x1<<" "<<x2<<" "<<y<< "\n";
	}
	
	
	myfile.close();
	int* pX1 = new int[DATA_SIZE]; 
	int* pX2 = new int[DATA_SIZE]; 
	int* pY = new int[DATA_SIZE]; 
	if(!readkNNData(pX1,pX2,pY,DATA_SIZE))
		cout<<"false returned";
	getchar();
	return 0;
}

bool readkNNData(int*pX1, int* pX2, int *pY, int dataSize)
{
	if( (NULL == pX1) ||
		(NULL == pX2) ||
		(NULL == pY) )
	{
		return false;
	}

	ifstream file ("kNNData.txt");
	if(!file.is_open())
	{
		return false;
	}
	int pCheckClassSum[3];
	for(int i=0;i<NUM_CLASSES;++i)
	{
		pCheckClassSum[i] = 0;
	}
	int x1,x2,y;
	for(int i=0; i< dataSize;++i)
	{
		file>>x1>>x2>>y;
		pX1[i] = x1;
		pX2[i] = x2;
		pY[i] = y;
		pCheckClassSum[y]++;
		cout<<"koushik::  x1="<<x1<<"  x2="<<x2<<"  y="<<y<<"\n";
		cout<<"koushik::  pX1[i]="<<pX1[i]<<" pX2[i]="<<pX2[i]<<"  pY[i]="<<pY[i]<<"\n";
	}
	for(int i=0;i<NUM_CLASSES;++i)
	{
		cout<<"pCheckClassSum["<<i<<"]="<<pCheckClassSum[i]<<"\n";
	}
	return true;
}