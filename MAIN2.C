#include <stdio.h>
#include <string.h>
#include <math.h>
#include<stdlib.h>
#include<time.h>
#include<limits.h>
double X2[20];
double z[20];
double Wji[20][20];
double Wkj[20][20];
double D_Wkj_ofNew[20][20];
double D_Wji_ofNew[20][20];
double D_Wkj[20][20];
double D_Wji[20][20];

//--------------------------------------------------------starting of forward propagation---------------------------------------------------
double sigmoid(double x){
	double y,exp_value;
	exp_value=exp(-x);
	y=1/(1+exp_value);
	return y;
}
double forward_propagation(double X1[],int n,int key){
	//-------------------------------------------------------sigmoid calculation for hiddenlayer-------------------------------------
	double X1_bias[20];
	X1_bias[0]=1.0;
	//int h_index;
	//double X2_bias;
	for(int j=1;j<17;j++){
		X1_bias[j]=X1[j-1];
	}
	for(int j=0;j<5;j++){
	    double sum=0;
		for(int i=0;i<17;i++){
			sum=sum+X1_bias[i]*Wji[i][j];
		}
		//printf("%lf ",sum);
		//printf("\n");
		X2[j]=sigmoid(sum);
	}
	/*for(int j=0;j<5;j++){
	    printf("%lf ",X2[j]);
	  }
	
	 printf("  hidden\n");*/
	//-------------------------------------------------------sigmoid calculation for output layer-----------------------------------
	double X2_bias[20];
	X2_bias[0]=1.0;
	for(int k=1;k<5;k++){
		X2_bias[k]=X2[k-1];
	}
	
	for(int k=0;k<10;k++){
	    double sum1=0;
		for(int j=0;j<6;j++){
		   sum1=sum1+X2_bias[j]*Wkj[j][k];  
		}
		//printf("%lf ",sum1);
		//printf("\n");
		z[k]=sigmoid(sum1);
	}
	if(key==1){
	float max=INT_MIN * 1.0;
	int index;
	for(int p=0;p<10;p++){
		      printf("%lf ",z[p]);
		   		if(z[p] > max){
		   			max = z[p];
		   			index = p;
		   		}
		   }
		   printf("%d\n",index);
        }		   
		   
	/*for(int j=0;j<10;j++){
		printf("%lf ",z[j]);
	 }
	 printf("  output\n");*/
}
//----------------------------------------------------------starting of backward propagation------------------------------------------
void backward_propagation(double t[], double X1[], double eta){
	    
        //---------------------------------------process between hidden and output layer---------------------------------------
	    double delta_output [20]; // error for the output layer
	    //updating the values of error in delta_output
	    for(int k=0; k<10; k++){
	    	delta_output[k] =  ( z[k]-t[k] );
		  }
		double X2_bias[20];
	    X2_bias[0]=1.0;
	    for(int k=1;k<5;k++){
		   X2_bias[k]=X2[k-1];
	    }
		for(int k=0; k<10; k++){
		   for(int j=0; j<6; j++){
		  	D_Wkj[j][k] = eta * X2_bias[j] * delta_output[k];
			}
		}
        //---------------------------------------process between input and hidden layer---------------------------------------
		double delta_hidden[10]; // error for the hidden layer
	    //updating the values of error in delta_hidden
		/*for(int j=1; j<6; j++){
			double sum = 0;
			for(int r=0; r<10; r++){
				sum = sum+( delta_output[r] * Wkj[j][r] * ( X2[j-1]*(1.0-X2[j-1]) ) );
			}
			delta_hidden[j-1] = sum;
		}*/
		
		//update of Delta weight between input and hidden layer  /_\Wji
        double X1_bias[20];
	    X1_bias[0]=1.0;
	    int h_index;
	    //double X2_bias;
	    for(int j=1;j<17;j++){
		    X1_bias[j]=X1[j-1];
	     }
		for(int j=0; j<5; j++){
			for(int i=0; i<17; i++){
				D_Wji[i][j] = eta*X1_bias[i]*delta_hidden[j];
			}
		}
}


// --------------------------------------------------starting of main function------------------------------------------------
double RandomNumber(float Min, float Max)
{
    return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
}
// calculating norm
double norm_calculation(int nrow, int ncol,double A[][ncol]){
	int i,j;
	double norm;
	double sum=0;
	for(i=0; i<nrow; i++){
		for(j=0; j<ncol; j++){
			sum = sum+ (A[i][j]*A[i][j]);
		}
	}
    norm=sqrt((double)sum);
    return (norm);    
}
int main(){
    //------------------------------------------------- random initialization to weights--------------------------------------------
    srand(time(NULL));
    for(int i=0; i<17; i++){
    	for(int j=0; j<5; j++){
    		Wji[i][j] = RandomNumber(0.01,0.05);
    		//D_Wji_ofNew[i][j] = Wji[i][j];
    	}
    }
    
    for(int i=0; i<17; i++){
    	for(int j=0; j<5; j++){
    		printf("%lf ", Wji[i][j]);
    	}
    	printf("\n");
    }
    printf("initial Wji weights\n");
    
    double norm_ofWji;
    norm_ofWji = norm_calculation(17,5,D_Wji_ofNew);
    //printf("%lf\n", norm_ofWji);
    for(int j=0; j<6; j++){
        for(int k=0; k<10; k++){
            Wkj[j][k] = RandomNumber(0.01,0.05);
            //D_Wkj_ofNew[j][k] = Wkj[j][k];
        }
    }
    
    for(int j=0; j<6; j++){
    	for(int k=0; k<10; k++){

    		printf("%lf ", Wkj[j][k]);

    	}
    	printf("\n");
    }
    printf("initial Wkj weights\n");
    //double norm_ofWkj = norm_calculation(6,10,D_Wkj_ofNew);
    //printf("%lf\n", norm_ofWkj);
    //double epsilon = 0.01;// assigning epsilon

    FILE *f = fopen("train1.txt", "r");
		double A[3000][50];
		while(getc(f)!= EOF){
			for(int i=0;i<2216;i++){
				for(int j=0;j<17;j++){
					fscanf(f,"%lf",&A[i][j]);
				}
			}
		}
			//double avg_norm = (norm_ofWji+norm_ofWkj)/2;
			//printf("%lf ", avg_norm);
			//printf("\n");
		
    //-------------------------------------------------the process for extracting the data and accesing--------------------------------------- 
    int epho=0;
	while( epho < 5000){
		
			//-----------------------------------------the process for every training example--------------------------------------------------
			for(int i=0; i< 2216; i++){
				double X[20];
				for(int j=1; j<17; j++){
					X[j-1] = A[i][j];
				}
				//----------------------------------------creation of  target output------------------------------------------------------------
				int class_label = (int)A[i][0];// assigning the label value
				//printf("%d \n", class_label);
				double T[15];
				for(int j=0; j<10; j++){
						T[j]=0.0;
					}	
				T[class_label-1]=1.0;
				//-----------------------------------------end of the process on target output-------------------------------------------------- 
				// calling the fuctions to calculate forward and  backward
				forward_propagation(X,5,0);
				backward_propagation(T,X,0.001);
				//-----------------------------------------process to summing the needed update weights of every example--------------------------------
				for(int l=0; l<17; l++){
                  for(int j=0; j<5; j++){
                      Wji[l][j] = Wji[l][j] + D_Wji[l][j];
                      }
                 }
				for(int j=0; j<6; j++){
                   for(int k=0; k<10; k++){
                      Wkj[j][k] = Wkj[j][k] + D_Wkj[j][k];// updating weights 
                     }
                  }
                  	
               
                /*for(int j=0; j<6; j++){
                	for(int k=0; k<10; k++){
                		D_Wkj_ofNew[j][k] = D_Wkj_ofNew[j][k] + D_Wkj[j][k];
                	}
                }   
                //updating values of weights of input to hidden
                for(int i=0; i<17; i++){
                	for(int j=0; j<5; j++){
                		D_Wji_ofNew[i][j] = D_Wji_ofNew[i][j] + D_Wji[i][j];
                	}
                }*/
                
			}
	      	
		//norm_ofWkj = norm_calculation(6,10,D_Wkj_ofNew);
		//norm_ofWji = norm_calculation(17,5,D_Wji_ofNew);
		//avg_norm = (norm_ofWji+norm_ofWkj)/2;
		//printf("%lf ", avg_norm);
		//printf("\n");
		//printf("%lf\n", norm_ofWkj);
		// udating the old weights of hidden and output 
        // udating the old weights of hidden and output 
        epho++; 
 
    }
    
	
	for(int i=0; i<17; i++){
    	    for(int j=0; j<5; j++){
    		printf("%lf ", Wji[i][j]);
    	}
    	   printf("\n");
	 }
	 printf("--------------------\n");
	 for(int j=0; j<6; j++){
                   for(int k=0; k<10; k++){
                      printf("%lf ", Wkj[j][k]);
                     }
                    printf("\n");
                  }
         printf("--------------------\n");
	 FILE *f1= fopen("test.txt", "r");
		double Atest[3000][50];
		while(getc(f1)!= EOF){
			for(int i=0;i<999;i++){
				for(int j=0;j<17;j++){
					fscanf(f1,"%lf",&Atest[i][j]);
				}
			}
		}
		int label[999];
		
		for(int i=0;i<999;i++){
		   double Xtest[20];
				for(int j=1; j<17; j++){
					Xtest[j-1] = Atest[i][j];
				}
		   forward_propagation(Xtest,5,1);
		   
		   //printf("----initial MAx:%lf\n",max);
		   //int index;
		   
		   
		   label[i]=index+1;
		   }
	 int W[17]={6,11,6,8,4,6,7,7,6,9,8,11,3,7,6,8};
	 
	 for(int i=0;i<999;i++){
	    printf("%d\n  ",label[i] );
	   }
	 printf("\n");   
	//printf("%lf\n", avg_norm);
	return 0;
}
