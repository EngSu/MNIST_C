//
//  fixedpoint_mnist_program
//
//  Created by Sumaia Atiwa on 2019-07-17.
//  Copyright © 2019 Sumaia Atiwa. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include<assert.h>
#include <sys/mman.h>
#include <sys/time.h>

#define N 784
#define H_NODES 2
#define O_NODES 2


int main()
{
    FILE *hidden_file0,*hidden_file1,*file,*fp0, *fp1;

    signed int myvariable0, myvariable1;// to read weights/parameters for the hidden layer and output layer

    signed int myvar; // to read the inputs/features




    signed int x_temp[2115][784];
    signed int w0[784], w1[784];
    signed int w00[2],w01[2];
    
    signed int sigmoid[2];
    signed int exp_val_out[2], sum_exp_val;
    signed int softmax[2];
    signed int mul_result0[784], mul_result1[784];
    signed int mul_result00[2],mul_result01[2];
    signed int acc_temp[2], abs_acc_temp[2];
    signed int out_temp[2], abs_out_temp[2];
    signed int out0[2115], out1[2115];
    signed int temp[2];

    int i,j;
    int ii =0;
    double secs = 0;
    struct timeval stop, start;
    //==================== reading the weights of the hidden layar =====================//
    hidden_file0 =fopen("hidden_wieghts_FP0.txt", "r");
    if(hidden_file0 == 0) {
        perror("fopen");
        printf("Error in hidden_layer_wieghts file\n");
        exit(1);
    }
    for(i = 0; i < N; i++)
    {
            fscanf(hidden_file0,"%d",&myvariable0);
            w0[i] = myvariable0 ;
    }
    
    hidden_file1 =fopen("hidden_wieghts_FP1.txt", "r");
    if(hidden_file1 == 0) {
        perror("fopen");
        printf("Error in hidden_layer_wieghts file\n");
        exit(1);
    }
    for(i = 0; i < N; i++)
    {
        fscanf(hidden_file1,"%d",&myvariable0);
        w1[i] = myvariable0 ;
    }
    //==================================================================================//
    
    //==================== reading the weights of the output layar =====================//
    
    fp0=fopen("layer2_wieghts_FP00.txt", "r");
    if(fp0 == 0) {
        perror("fopen");
        printf("Error in layer2_wieghts file\n");
        exit(1);
    }
    for(i = 0; i < 2; i++)
    {
        
            fscanf(fp0,"%d",&myvariable1);
            w00[i] = myvariable1 ;
    }
    
    fp1=fopen("layer2_wieghts_FP01.txt", "r");
    if(fp1 == 0) {
        perror("fopen");
        printf("Error in layer2_wieghts file\n");
        exit(1);
    }
    for(i = 0; i < 2; i++)
    {
        fscanf(fp1,"%d",&myvariable1);
        w01[i] = myvariable1 ;
    }
    //==================================================================================//
    file = fopen("inputs_FP.txt", "r");
    if(file == 0) {
        perror("fopen");
        printf("Error in inputs file\n");
        exit(1);
    }
  
    for(i = 0; i < 2115; i++)
    {
        for (j = 0 ; j < N; j++)
        {
            fscanf(file,"%d",&myvar);
            x_temp[i][j] = myvar ;
        }
    }
    
    //==================================================================================//
 
    for(ii=0;ii<2115;ii++) {

        gettimeofday(&start, NULL);
        acc_temp[0] = 0;
        acc_temp[1] = 0;
        out_temp[0] = 0;
        out_temp[1] = 0;

        for (int i = 0; i < N; i++){
            mul_result0[i] = w0[i] *  x_temp[ii][i];
            acc_temp[0] = acc_temp[0] +  mul_result0[i];
            mul_result1[i] = w1[i] * x_temp[ii][i];
            acc_temp[1] = acc_temp[1] + mul_result1[i];}
        
        for (j =0; j<H_NODES; j++){
            
            if (acc_temp[j] >= 0) {
                if (acc_temp[j] == 0) sigmoid[j]  = 64;
                else if (acc_temp[j] >= 131072) sigmoid[j] = 127;
                else if (acc_temp[j]< 131072 && acc_temp[j] >= 114688){
                    sigmoid[j] = 127;
                }
                
                else if (acc_temp[j] < 114688 && acc_temp[j] >= 98304 ) {
                    sigmoid[j] = 127;
                }
                
                else if (acc_temp[j] < 98304 && acc_temp[j] >= 81920 ) {
                    sigmoid[j] = 127;
                }
                
                else if (acc_temp[j] < 81920 && acc_temp[j] >= 65536) {
                    sigmoid[j] = 126;
                }
                
                
                else if (acc_temp[j] < 65536 && acc_temp[j] >= 49152) {
                    sigmoid[j] = 125;
                }
                
                else if (acc_temp[j] < 49152 && acc_temp[j] >=32768) {
                    sigmoid[j] = 119;
                }
                
                else if (acc_temp[j] < 32768 && acc_temp[j] >=16384) {
                    sigmoid[j] = 106;
                }
                
                else if (acc_temp[j] < 16384 && acc_temp[j] >=0) {
                    sigmoid[j] = 80;
                }
            }
            else {
                
                abs_acc_temp[j] = - acc_temp[j];
                if (abs_acc_temp[j] >= 131072) sigmoid[j] = 0;
                else if (abs_acc_temp[j]< 131072 && abs_acc_temp[j] >= 114688){
                    sigmoid[j] = 0;
                }
                
                else if (abs_acc_temp[j] < 114688 && abs_acc_temp[j] >= 98304 ) {
                    sigmoid[j] = 0;
                }
                
                else if (abs_acc_temp[j] < 98304 && abs_acc_temp[j] >= 81920 ) {
                    sigmoid[j] = 0;
                }
                
                else if (abs_acc_temp[j] < 81920 && abs_acc_temp[j] >= 65536) {
                    sigmoid[j] = 1;
                }
                
                
                else if (abs_acc_temp[j] < 65536 && abs_acc_temp[j] >= 49152) {
                    sigmoid[j] = 2;
                }
                
                else if (abs_acc_temp[j] < 49152 && abs_acc_temp[j] >=32768) {
                    sigmoid[j] = 8;
                }
                
                else if (abs_acc_temp[j] < 32768 && abs_acc_temp[j] >=16384) {
                    sigmoid[j] = 21;
                }
                
                else if (abs_acc_temp[j] < 16384 && abs_acc_temp[j] >=0) {
                    sigmoid[j] = 47;
                }
                
            }

        }

        //printf("case %d: %d        %d\n",ii, sigmoid[0], sigmoid[1]);

        for ( i = 0; i < 2; i++){
            mul_result00[i] = w00[i] * sigmoid[i];
            out_temp[0] = out_temp[0] + mul_result00[i];
            mul_result01[i] = w01[i] * sigmoid[i];
            out_temp[1] = out_temp[1] + mul_result01[i];}
 
        sum_exp_val =0;
        for (j =0; j<O_NODES; j++){
          
        if (out_temp[j] == 0) exp_val_out[j] = 127;
        else {
            if (out_temp[j] >= 0) {
                
                if (out_temp[j] <= 14745 && out_temp[j] > 12779 ) {
                    exp_val_out[j] = 78;
                }
                
                else if (out_temp[j] <= 12779 && out_temp[j] > 10813 ) {
                    exp_val_out[j] = 69;
                }
                
                else if (out_temp[j] <= 10813 && out_temp[j] > 8847 ) {
                    exp_val_out[j] = 61;
                }
                
                else if (out_temp[j] <= 8847 && out_temp[j] > 6881 ) {
                    exp_val_out[j] = 54;
                }
                
                else if (out_temp[j] <= 6881 && out_temp[j] > 4915 ) {
                    exp_val_out[j] = 48;
                }
                
                else if (out_temp[j] <= 4915 && out_temp[j] > 2949 ) {
                    exp_val_out[j] = 43;
                }
                
                else if (out_temp[j] <= 2949 && out_temp[j] > 983 ) {
                    exp_val_out[j] = 38;
                }
                
                else if (out_temp[j] <= 983 && out_temp[j] > 0 ) {
                    exp_val_out[j] = 33;
                }
            }
            else {
                abs_out_temp[j] = - out_temp[j];
                if (abs_out_temp[j] <= 14745 && abs_out_temp[j] > 12779 ) {
                    exp_val_out[j] = 13;
                }
                else if (abs_out_temp[j] <= 12779 && abs_out_temp[j] > 10813 ) {
                    exp_val_out[j] = 14;
                }
                else if (abs_out_temp[j] <= 10813 && abs_out_temp[j] > 8847 ) {
                    exp_val_out[j] = 16;
                }
                
                else if (abs_out_temp[j] <= 8847 && abs_out_temp[j] > 6881 ) {
                    exp_val_out[j] = 18;
                }
                
                else if (abs_out_temp[j] <= 6881 && abs_out_temp[j] > 4915 ) {
                    exp_val_out[j] = 21;
                }
                
                else if (abs_out_temp[j] <= 4915 && abs_out_temp[j] > 2949 ) {
                    exp_val_out[j] = 23;
                }
                
                else if (abs_out_temp[j] <= 2949 && abs_out_temp[j] > 983 ) {
                    exp_val_out[j] = 26;
                }
                
                else if (abs_out_temp[j] <= 983 && abs_out_temp[j] > 0 ) {
                    exp_val_out[j] = 30;
                }
            }
     }
                    

}
        sum_exp_val =  exp_val_out[0] + exp_val_out[1];
        softmax[0] = (exp_val_out[0]<<7) / sum_exp_val;
        softmax[1] = (exp_val_out[1]<<7) / sum_exp_val;

        printf("Output %d: %d       %d \n",ii, softmax[0], softmax[1]);

        gettimeofday(&stop, NULL);
    

    }
    
    secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
    
    printf("=====================================================================\
    \n Time Taken(sec) for Software-based MNIST: %f Secs\n=====================================================================\n\
    \n", secs);
    
    fclose(file);
    fclose(hidden_file0);
    fclose(hidden_file1);
    fclose(fp0);
    fclose(fp1);
    return 0;
}

