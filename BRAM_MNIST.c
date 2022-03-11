#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/time.h>



// The purpose this test is to show that users can get to devices in user
// mode for simple things like GPIO. This is not to say this should replace
// a kernel driver, but does provide some short term solutions sometimes
// or a debug solution that can be helpful.

// This test maps a GPIO in the hardware into the user space such that a
// GPIO signal can be toggled fast. On the ML507 reference system, the 
// signal could be toggled about every 50 ns which is pretty fast.

// This test was derived from devmem2.c.


#define BRAM_CTRL_ADDRESS_0 	0x40000000
#define BRAM_CTRL_ADDRESS_1 	0x43C00000
#define SIMD_LITE_ADDRESS_0 	0x80000000
#define SIMD_FULL_ADDRESS_1 	0x8AA00000
#define GPIO_DATA_OFFSET 	0
#define GPIO_DIRECTION_OFFSET 	4

#define NO_LABELS	2115
 
#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
int close(int fd);


double round_f(float var) 
{ 
    float value = (int)(var * 1000000 + .5); 
    return (double)value / 1000000; 
} 
int main() 
{   

        FILE *file,*outfile, *fp_output, *hiddenweights, *outputweights, *errfile0, *errfile1, *labelfile, *accfile0, *accfile1, *hw_labels, *sw_labels, *actual_labels, *timefile;
        double outvariable;
	int labelvariable;
        int myvar;
	int hw_correct_out = 0;
	int sw_correct_out = 0;
	int actual_label;
	int predicted_label;
	int sw_label;
	float hw_accuracy;
        float sw_accuracy;
        float abs_avg_err0= 0, abs_avg_err1=0;
        float avg_err0= 0, avg_err1=0;
        double secs = 0;
        double exp_val_out[2], sum_exp_val, softmax[2];
        double x[784];
        int out_temp[2];
        double acc_out0, acc_out1;
        double out0[2115], out1[2115];
	double temp[2];
	int i,j;
	int ii =0;
	int jj =0;
	int x_temp[2115][784];
	int x_in[196];
        int hidden_weights[196];
        int output_weights[2];
        struct timeval stop, start;
	int memfd;
        void *bram_base0, *mapped_bram_base0; 

        void *bram_base1, *mapped_bram_base1; 

 
	void *mapped_base0, *mapped_dev_base0; 
	off_t dev_base0 = SIMD_LITE_ADDRESS_0;
        void *mapped_base1, *mapped_dev_base1; 
	off_t dev_base1 = SIMD_FULL_ADDRESS_1;

        hiddenweights=fopen("hidden_weights.txt", "r");
        outputweights=fopen("output_weights.txt", "r");

        memfd = open("/dev/mem", O_RDWR | O_SYNC);
    	if (memfd == -1) {
		printf("Can't open /dev/mem.\n");
		exit(0);
	}
	//printf("/dev/mem opened.\n"); 
    
	// Map one page of memory into user space such that the device is in that page, but it may not
	// be at the start of the page

	bram_base0 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, BRAM_CTRL_ADDRESS_0 & ~MAP_MASK);
    	if (mapped_base0 == (void *) -1) {
		printf("Can't map the memory to user space.\n");
		exit(0);
	}

        bram_base1 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, BRAM_CTRL_ADDRESS_1 & ~MAP_MASK);
    	if (mapped_base1 == (void *) -1) {
		printf("Can't map the memory to user space.\n");
		exit(0);
	}
 	//printf("Memory mapped at address %p.\n", mapped_base); 
 
	// get the address of the device in user space which will be an offset from the base 
	// that was mapped as memory is mapped at the start of a page 
   
	mapped_bram_base0 = bram_base0 + (BRAM_CTRL_ADDRESS_0 & MAP_MASK);
        mapped_bram_base1 = bram_base1 + (BRAM_CTRL_ADDRESS_1 & MAP_MASK);
      
for (j = 0 ; j < 196; j++)
        {
            fscanf(hiddenweights,"%d",&myvar);
            hidden_weights[j] = myvar ;
//printf("%d:	%d \n",j, hidden_weights[j]);
        }

for (j = 0 ; j < 2; j++)
        {
            fscanf(outputweights,"%d",&myvar);
            output_weights[j] = myvar ;
//printf("%d:	%d \n",j, output_weights[j]);
        }


       memcpy((mapped_bram_base0 + GPIO_DATA_OFFSET),hidden_weights,4*196);

        memcpy((mapped_bram_base1 + GPIO_DATA_OFFSET),output_weights,2);



       if (munmap(bram_base0, MAP_SIZE) == -1) {
		printf("Can't unmap memory from user space.\n");
		exit(0);
	}
	if (munmap(bram_base1, MAP_SIZE) == -1) {
		printf("Can't unmap memory from user space.\n");
		exit(0);
	}
	
	close(memfd);


  
	
	memfd = open("/dev/mem", O_RDWR | O_SYNC);
    	if (memfd == -1) {
		printf("Can't open /dev/mem.\n");
		exit(0);
	}
	//printf("/dev/mem opened.\n"); 
    
	// Map one page of memory into user space such that the device is in that page, but it may not
	// be at the start of the page

	mapped_base0 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, dev_base0 & ~MAP_MASK);
    	if (mapped_base0 == (void *) -1) {
		printf("Can't map the memory to user space.\n");
		exit(0);
	}

        mapped_base1 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, dev_base1 & ~MAP_MASK);
    	if (mapped_base1 == (void *) -1) {
		printf("Can't map the memory to user space.\n");
		exit(0);
	}
 	//printf("Memory mapped at address %p.\n", mapped_base); 
 
	// get the address of the device in user space which will be an offset from the base 
	// that was mapped as memory is mapped at the start of a page 
   
	mapped_dev_base0 = mapped_base0 + (dev_base0 & MAP_MASK);
        mapped_dev_base1 = mapped_base1 + (dev_base1 & MAP_MASK);

	*((volatile unsigned long *) (mapped_dev_base0 + GPIO_DIRECTION_OFFSET)) = 0;

	// toggle the output as fast as possible just to see how fast it works

              

  // send in_valid signal 0
	//	*((volatile unsigned long *) (mapped_dev_base + GPIO_DATA_OFFSET)) = 0;

              // 784 inputs of 8 bits
//==================== reading the inputs of the hidden layar =====================//   
 
    file = fopen("inputs_FP.txt", "r");
    if(file == NULL) {
        perror("Error opening file");
        return(-1);
    }

	for(i = 0; i < 2115; i++)
   	 {
        for (j = 0 ; j < 784; j++)
        {
            fscanf(file,"%d",&myvar);
            x_temp[i][j] = myvar ;
//printf("%d:	%d \n",i, x_temp[i][j]);
        }
     // printf("\n\n");
    }

    
    
//==================================================================================// 

 	outfile=fopen("outputs.txt", "r");
        fp_output=fopen("param_fp_outputs.txt", "w");
        errfile0=fopen("new_errfile0.txt", "w");
        errfile1=fopen("new_errfile1.txt", "w");
        accfile0=fopen("hw_acc.txt", "w");
 	accfile1=fopen("sw_acc.txt", "w");
        sw_labels=fopen("sw_label.txt", "w");
        hw_labels=fopen("hw_label.txt", "w");
        actual_labels=fopen("actual_label.txt", "w");
        labelfile=fopen("labels.txt", "r");
        timefile=fopen("time_file.txt", "w");
        jj =0;
	hw_correct_out = 0;
	sw_correct_out = 0;
        avg_err0= 0, avg_err1=0;

	for(jj=0;jj<2115;jj++) {
        // send in_valid signal 0
	*((volatile unsigned long *) (mapped_dev_base0 + GPIO_DATA_OFFSET)) = 0;
        int k =0;
        for (j = 0 ; j < 196; j++)
        {
          x_in[j] = (x_temp[jj][k+3]<<24)|(x_temp[jj][k+2]<<16)|(x_temp[jj][k+1]<<8)|(x_temp[jj][k]) ;
          k= k+4;
//printf("%d:	%d \n",j, x_in[j]);
        }

	gettimeofday(&start, NULL);
        memcpy((mapped_dev_base1 + GPIO_DATA_OFFSET),x_in,4*196);

      //  gettimeofday(&stop, NULL);
//if (jj == 0){
//for(int i = 0; i < 196; i=i+1){ 


 //printf("New Read : 0x%d   %ld \n\r", i, *((unsigned long *) (mapped_dev_base1 + GPIO_DATA_OFFSET +(4*i))));
//}
  //}   
        // send in_valid signal 1
	*((volatile unsigned long *) (mapped_dev_base0 + GPIO_DATA_OFFSET)) = 1;
	gettimeofday(&start, NULL);

	// read outputs 
	while ( *((volatile unsigned long *) (mapped_dev_base0 + GPIO_DATA_OFFSET +12)) == 0 ){}
        gettimeofday(&stop, NULL);

secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
 fprintf(timefile, "%d	%f\n",(jj+1), secs); 

  // printf("output ready  :%d \n\r", *((unsigned long *) (mapped_dev_base + GPIO_DATA_OFFSET+ 12)));


             	 out_temp[0] =  *((volatile signed int *) (mapped_dev_base0 + GPIO_DATA_OFFSET +4)) ;
            	 out_temp[1] =  *((volatile signed int *) (mapped_dev_base0 + GPIO_DATA_OFFSET +8)) ;

	     // printf("output%d:	%.6f	%.6f \n\n\r",jj, (double)(*((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 4)))/127, (double)(*((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 8)))/127);
           //  printf("output%d:	%d	%d \n\n\r",jj, *((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 4)), *((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 8)));
	     // printf("output 2  :%.6f \n\r", );
            // for (j =0; j<2; j++){
		//exp_val_out[j] = exp(out_temp[j]);
		//sum_exp_val = sum_exp_val + ((double)out_temp[j]);
		//printf("Y_node%d = %f \n\n",j, sum_exp_val);
       		//}
	  	acc_out0 = round_f((float)(*((volatile signed int *) (mapped_dev_base0 + GPIO_DATA_OFFSET+ 4)))/128);
	  	acc_out1 = round_f((float)(*((volatile signed int *) (mapped_dev_base0 + GPIO_DATA_OFFSET+ 8)))/128);


	//acc_out0 = (float)(*((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 4)))>>7;
	//acc_out1 = (float)(*((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 8)))>>7;
      //printf("Hardware Output %d: %.6f		%.6f \n",jj, acc_out0, acc_out1);
       fprintf(fp_output,"%d	%d\n", *((volatile signed int *) (mapped_dev_base0 + GPIO_DATA_OFFSET+ 4)),*((volatile signed int *) (mapped_dev_base0 + GPIO_DATA_OFFSET+ 8)));
      // printf(" output %d:	%d	%d\n",jj, *((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 4)),*((volatile signed int *) (mapped_dev_base + GPIO_DATA_OFFSET+ 8)));
	for (j = 0 ; j < 2; j++){
	      fscanf(outfile,"%lf",&outvariable);
	      temp[j] = outvariable ;
	    }
		out0[jj] = round_f(temp[0]);
		out1[jj] = round_f(temp[1]);
		//printf("Actual Output %d: %.6f	%.6f \n",jj, out0[jj], out1[jj]);
		//printf("Error %d: %.6f	%.6f \n",jj, out0[jj]-acc_out0, out1[jj]-acc_out1);
                if (out0[jj]>acc_out0)
		{
		abs_avg_err0 = out0[jj] - acc_out0;
		}
                else abs_avg_err0 = acc_out0 - out0[jj];

                if (out1[jj]>acc_out1)
		{
		abs_avg_err1 = out1[jj] - acc_out1;
		}
                else abs_avg_err1 = acc_out1 - out1[jj];

		avg_err0 = avg_err0 +  (abs_avg_err0/out0[jj]);
		avg_err1 = avg_err1 +  (abs_avg_err1/out1[jj]);
                fprintf(errfile0,"%d	%.6f\n",jj, (out0[jj]-acc_out0));
                fprintf(errfile1,"%d	%.6f\n",jj, (out1[jj]-acc_out1));
                
		
		fscanf(labelfile,"%d",&labelvariable);
		actual_label = labelvariable;
                printf("Actual Label %d:	{%d}	\n",(jj+1), actual_label);
                if (acc_out0 > acc_out1){
		predicted_label = 0;
		}
		else
		{
		predicted_label = 1;
		}
	        fprintf(actual_labels, "%d	%d\n", jj, actual_label); 
                fprintf(hw_labels, "%d	%d\n", jj, predicted_label); 
                printf("Predicted Label %d:	{%d}	\n",(jj+1), predicted_label);
		if (predicted_label == actual_label )
		{
		hw_correct_out = hw_correct_out+1;
		}

                hw_accuracy = ((float)(hw_correct_out) / (float)(jj+1)) * 100;
		printf("Hardware Accuracy = (%d/%d) * 100 =	%.2f%	\n",hw_correct_out,(jj+1), hw_accuracy);
                fprintf(accfile0, "%d	%f\n",(jj+1), hw_accuracy); 

                if (out0[jj] > out1[jj]){
		sw_label = 0;
		}
		else
		{
		sw_label = 1;
		}
                fprintf(sw_labels, "%d	%d\n", jj, sw_label); 
 		if (sw_label == actual_label )
		{
		sw_correct_out = sw_correct_out+1;
		}
                sw_accuracy = ((float)(sw_correct_out) / (float)(jj+1)) * 100;
		printf("Software Accuracy = (%d/%d) * 100 =	%.2f%	\n\n\n",sw_correct_out,(jj+1), sw_accuracy);
                fprintf(accfile1, "%d	%f\n",(jj+1), sw_accuracy); 

                if (predicted_label != sw_label)
		{
		 printf("unmatched label#: %d\n",(jj+1));
		}
}
		avg_err0 = (avg_err0/2115) * 100;
		avg_err1 = (avg_err1/2115) * 100;
		//printf("Average Error of Label 0: %.2f%\n",avg_err0);
		//printf("Average Error of Label 1: %.2f%\n",avg_err1);

	//printf("took %lu\n", stop.tv_usec - start.tv_usec);

//	secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

 // printf("=========================================================================\
\n Time Taken(sec) for accelerated-based MNIST with Full-AXI: %f\n=========================================================================\n\
\n",secs);
	

	// unmap the memory before exiting
 	
	if (munmap(mapped_base0, MAP_SIZE) == -1) {
		printf("Can't unmap memory from user space.\n");
		exit(0);
	}
	if (munmap(mapped_base1, MAP_SIZE) == -1) {
		printf("Can't unmap memory from user space.\n");
		exit(0);
	}
	
	close(memfd);
        fclose(file);
        fclose(outfile);
        fclose(hiddenweights);
        fclose(outputweights);
        fclose(fp_output);
	fclose(errfile0);
	fclose(errfile1);
	fclose(labelfile);
	fclose(accfile0);
	fclose(accfile1);
	fclose(actual_labels);
	fclose(hw_labels);
	fclose(sw_labels);
	fclose(timefile);
	return 0;
}
