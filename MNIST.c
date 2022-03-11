#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include<assert.h>
#include <sys/mman.h>
#include <sys/time.h>

#define N 784
#define H_NODES 4
#define O_NODES 2
#define WIDTH  4

double round_f(float var) 
{ 
    float value = (int)(var * 1000000 + .5); 
    return (double)value / 1000000; 
} 
int main() 
{
        	FILE *myfile,*file,*fp, *outfile;
       		double myvariable0, myvariable1, outvariable;
		double secs = 0;
		double w0[784], w1[784],w2[784],w3[784];
		double w00[4],w01[4];
       		double w_temp[4];
       		double w00_temp[2];
		double exp_val_hidden[4], sigmoid[4];
		double exp_val_out[2], sum_exp_val, softmax[2];
		double mul_result0[784], mul_result1[784],  mul_result2[784],  mul_result3[784];
                double mul_result00[4],mul_result01[4];
		double acc_temp[4];
		double out_temp[2];
                double out0[2115], out1[2115];
		double temp[2];
		char chr;
		char num[10];
		int index =0;
		int i,j;
		int ii =0;
		int jj =0;
		int ** matrix;
		struct timeval stop, start;
//==================== reading the weights of the hidden layar =====================//
    myfile=fopen("weights.txt", "r");
    for(i = 0; i < N; i++)
    {
        for (j = 0 ; j < WIDTH; j++)
        {
            fscanf(myfile,"%lf",&myvariable0);
            w_temp[j] = myvariable0 ;
        }
        w0[i] = w_temp[0];
        w1[i] = w_temp[1];
        w2[i] = w_temp[2];
        w3[i] = w_temp[3];
 printf("%.9f \n", w3[i]);
    }
//==================================================================================// 

//==================== reading the weights of the output layar =====================//

  fp=fopen("layer2.txt", "r");
  for(i = 0; i < 4; i++)
  {
    for (j = 0 ; j < 2; j++)
    {
      fscanf(fp,"%lf",&myvariable1);
      w00_temp[j] = myvariable1 ;
    }
   w00[i] = w00_temp[0];
   w01[i] = w00_temp[1];
 // printf("%.9f \n", w01[i]);
}
//==================================================================================// 
//==================== reading the inputs of the hidden layar =====================//   
    matrix = (int** )malloc(sizeof(int*) *2115);
    assert(matrix);
    for(i=0;i<2115;i++) {
        matrix[i] = (int* )malloc(sizeof(int)*784);
        assert(matrix[i]);
    }
    for(i=0;i<2115;i++){
        for(j=0;j<784;j++) {
            matrix[i][j] = 0;
        }
    }
    
    file = fopen("inputs.csv", "r");
    if(file == NULL) {
        perror("Error opening file");
        return(-1);
    }
    while( (chr=fgetc(file)) !=EOF ){
        
        if(chr==','){
            index=0;
            //printf("%d ",atoi(num));
            matrix[ii][jj] = atoi(num);
            //printf("ii=%d, jj=%d \n",ii,jj);
            jj++;
            
            for(i=0;i<10;i++) {
                num[i]='\0';
            }
            continue;
        }
        else if(chr=='\n'){
           // printf("ii=%d, jj=%d \n",ii,jj);
            ii++;
            jj=0;
            //printf("\n");
        }
        else {
            num[index++] = chr;
        }       
    }
//==================================================================================// 
    outfile=fopen("outputs.txt", "r");
    for(ii=0;ii<2115;ii++) {
        
        acc_temp[0] = 0;
        acc_temp[1] = 0;
        acc_temp[2] = 0;
        acc_temp[3] = 0;
	out_temp[0] = 0;
        out_temp[1] = 0;
        gettimeofday(&start, NULL);

	for (int i = 0; i < N; i++){
		mul_result0[i] = w0[i] * ((double) matrix[ii][i]);
		acc_temp[0] = acc_temp[0] +  mul_result0[i];
		mul_result1[i] = w1[i] * ((double) matrix[ii][i]);
		acc_temp[1] = acc_temp[1] + mul_result1[i];
		mul_result2[i] = w2[i] * ((double) matrix[ii][i]);
		acc_temp[2] = acc_temp[2] + mul_result2[i];
     		mul_result3[i] = w3[i] * ((double) matrix[ii][i]);
		acc_temp[3] = acc_temp[3] + mul_result3[i];
		//printf(" w = %.9f ---  x = %d ---  w * x =%.9f  \n",w3[i], matrix[ii][i],mul_result3[i]);
	}
	for (j =0; j<H_NODES; j++){
		exp_val_hidden[j] = exp(-acc_temp[j]);
		sigmoid[j] = 1 / (1 + exp_val_hidden[j]);
		//printf("Y_node%d = %f \n\n",j, sigmoid[j]);
       		}
	for ( i = 0; i < 4; i++){
		mul_result00[i] = w00[i] * sigmoid[i];
		out_temp[0] = out_temp[0] + mul_result00[i];
		mul_result01[i] = w01[i] * sigmoid[i];
		out_temp[1] = out_temp[1] + mul_result01[i];
		//printf(" w = %.9f ---  x = %f ---  w * x =%.9f  \n",w00[i], sigmoid[i],out_temp[0]);
	}
	sum_exp_val =0;
	for (j =0; j<O_NODES; j++){
		exp_val_out[j] = exp(out_temp[j]);
		sum_exp_val = sum_exp_val + exp_val_out[j];
		//printf("Y_node%d = %f \n\n",j, sum_exp_val);
       		}
	softmax[0] = round_f(exp_val_out[0] / sum_exp_val);
	softmax[1] = round_f(exp_val_out[1] / sum_exp_val);
       printf("Output %d: %.6f		%.6f \n",ii, softmax[0], softmax[1]);

	for (j = 0 ; j < 2; j++){
	      fscanf(outfile,"%lf",&outvariable);
	      temp[j] = outvariable ;
	    }
		out0[ii] = round_f(temp[0]);
		out1[ii] = round_f(temp[1]);
		printf("Error %d: %.6f	%.6f \n",ii, out0[ii]-softmax[0], out1[ii]-softmax[1]);

    }

	gettimeofday(&stop, NULL);
	//printf("took %lu\n", stop.tv_usec - start.tv_usec);

	secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

printf("=================================================================================\
\n Time Taken(sec) for Software-based MNIST: %f\n====================================================================================\n\
\n", secs);

       for(i=0;i<2115;i++) {
           free(matrix[i]);
       }
       free(matrix);
       fclose(file);
       fclose(myfile);
       fclose(fp);
       fclose(outfile);

	return 0;
}
