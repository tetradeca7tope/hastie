#include "mex.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
//#include <tchar.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string.h>

typedef struct container{
	int idx;
	double value;
} sort_container;

bool operator<(const container& left, const container& right)
{
  return left.value < right.value;
}

#define feature(i,j) xp[(i)+(j)*dataNum]
#define target(i,j) tp[(i)+(j)*dataNum]


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // prhs[0] X
    // prhs[1] T
    // prhs[2] A ratio of randomly chosen features that are considered. Use less than 1 if you want to speed up training but the performance may deteriorate.
      
    
    // convert input args
    int featureDim = mxGetN( prhs[0] );
    int dataNum = mxGetM( prhs[0] );
    int targetDim = mxGetN( prhs[1] );
    
	double* xp = (double*)mxGetData(prhs[0]);
    double* tp = (double*)mxGetData(prhs[1]);
           
    double randomFeatureRatio = mxGetScalar(prhs[2]); 
    int selectFeatureDim = (int)(featureDim*randomFeatureRatio);
    
    // define variables    
    std::vector<double> bestSSE( selectFeatureDim, DBL_MAX );
    std::vector<double> bestThr( selectFeatureDim, 0 );
    std::vector<std::vector<double> > rightOutput( selectFeatureDim );
    for( int i=0; i<selectFeatureDim; i++ ){
        rightOutput[i].resize( targetDim, 0 );
    }
    std::vector<std::vector<double> > leftOutput( selectFeatureDim );
    for( int i=0; i<selectFeatureDim; i++ ){
        leftOutput[i].resize( targetDim, 0 );
    }
    
    std::vector<double> bestLeftSSE( selectFeatureDim, 0 );
    std::vector<double> bestRightSSE( selectFeatureDim, 0 );
    
    //////////////////////////////////////
    std::vector<int> randIdxes( featureDim );
    for( int i=0; i<featureDim; i++ ){
        randIdxes[i] = i;
    }
    std::random_shuffle ( randIdxes.begin(), randIdxes.end() );

    // main algorithm
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic )
#endif
    for( int i=0; i<selectFeatureDim; i++ ){
        int m = randIdxes[i]; // current dim
        
        std::vector<sort_container> sortByValue;
        sortByValue.resize( dataNum );
        for( int j=0; j<dataNum; j++ ){
            sortByValue[j].idx = j;
            sortByValue[j].value = feature(j,m);
        }
        std::sort(sortByValue.begin(),sortByValue.end());
        
        // calc initial average
        std::vector<double> aveLeft( targetDim, 0 );
        std::vector<double> aveRight( targetDim, 0 );
        std::vector<double> aveLeftPre( targetDim, 0 );
        std::vector<double> aveRightPre( targetDim, 0 );

        for( int j=0; j<dataNum; j++ ){
            for( int k=0; k<targetDim; k++ ){
                aveRight[k] += target(j,k);
            }
        }
        for( int j=0; j<targetDim; j++ ){
            aveRight[j] = aveRight[j] / dataNum;
        }

        // calc initial sse
        double sseLeft = 0;
        double sseRight = 0;
        
        int sizeLeft = 0;
        int sizeRight = 0;
        
        double sse;
        double thr;

        for( int j=0; j<dataNum; j++ ){
            for( int k=0; k<targetDim; k++ ){
                sseRight += ( target(j,k) - aveRight[k] ) * ( target(j,k) - aveRight[k] );
            }
        }

        for( int j=0; j<dataNum-1; j++ ){

            //remember ave
            aveLeftPre = aveLeft;
            aveRightPre = aveRight;

            //calc ave
            sizeLeft=j+1;
            sizeRight=dataNum-sizeLeft;

            for( int k=0; k<targetDim; k++ ){
                aveLeft[k]=(aveLeftPre[k]*(sizeLeft-1)+target(sortByValue[j].idx,k)) / sizeLeft;
                aveRight[k]=(aveRightPre[k]*(sizeRight+1)-target(sortByValue[j].idx,k)) / sizeRight;
            }

            for( int k=0; k<targetDim; k++ ){
                sseLeft = sseLeft + ( aveLeft[k] - aveLeftPre[k] ) * ( aveLeft[k] - aveLeftPre[k] ) * ( sizeLeft - 1 ) + ( target(sortByValue[j].idx,k) - aveLeft[k] ) * ( target(sortByValue[j].idx,k) - aveLeft[k] );
                sseRight = sseRight - ( aveRightPre[k] - aveRight[k] ) * ( aveRightPre[k] - aveRight[k] ) * sizeRight - ( target(sortByValue[j].idx,k) - aveRightPre[k] ) * ( target(sortByValue[j].idx,k) - aveRightPre[k] );
            }

            sse = sseLeft + sseRight;

            //calc thr
            if ( sortByValue[j].value != sortByValue[j+1].value ){
                thr = ( sortByValue[j].value + sortByValue[j+1].value ) / 2; //threshold
                if( sse < bestSSE[i] ){	
                    bestSSE[i] = sse;
                    bestThr[i] = thr;
                    leftOutput[i] = aveLeft;
                    rightOutput[i] = aveRight;
                    bestLeftSSE[i] = sseLeft;
                    bestRightSSE[i] = sseRight;
                }
            }            
        }
    }
    
    // determine the best dimension
    double  finalSSE = DBL_MAX;
    double finalLeftSSE;
    double finalRightSSE;
    int finalIdx;
    double finalThr;
    std::vector<double> finalLeftOutput( targetDim, 0 );
    std::vector<double> finalRightOutput( targetDim, 0 );
    
    for( int i=0; i<selectFeatureDim; i++ ){
        int m = randIdxes[i]; // current dim
        
        if( finalSSE > bestSSE[i] ){
            finalSSE = bestSSE[i];
            finalIdx = m;
            finalThr = bestThr[i];
            for( int j=0; j<targetDim; j++ ){
                finalLeftOutput[j] = leftOutput[i][j];
                finalRightOutput[j] = rightOutput[i][j];
            }
            finalLeftSSE = bestLeftSSE[i];
            finalRightSSE = bestRightSSE[i];            
        }
    }
        
    // Set output
    plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *p0 = (int*)mxGetData(plhs[0]);
    p0[0] = finalIdx + 1;
    
    plhs[1] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);  
    double *p1 = mxGetPr(plhs[1]);
    p1[0] = finalThr;
       
    plhs[2] = mxCreateNumericMatrix(1, targetDim, mxDOUBLE_CLASS, mxREAL);  
    double *p2 = mxGetPr(plhs[2]);
    for( int i=0; i<targetDim; i++ ){
        p2[i] = finalLeftOutput[i];
    }
        
    plhs[3] = mxCreateNumericMatrix(1, targetDim, mxDOUBLE_CLASS, mxREAL);  
    double *p3 = mxGetPr(plhs[3]);
    for( int i=0; i<targetDim; i++ ){
        p3[i] = finalRightOutput[i];
    }
    
    plhs[4] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    double *p4 = mxGetPr(plhs[4]);
    p4[0] = finalLeftSSE;
       
    plhs[5] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);  
    double *p5 = mxGetPr(plhs[5]);
    p5[0] = finalRightSSE;
}
    
        
                
