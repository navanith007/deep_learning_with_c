#include <stdio.h>
#include <string.h>
#include <math.h>
#include<stdlib.h>
#include<time.h>
#include<limits.h>



//--------------------------------------------------------starting of forward propagation---------------------------------------------------
float sigmoid(float x){
	float y,exp_value;
	exp_value=exp(-x);
	y=1/(1+exp_value);
	return y;
}
float forward_propagation(float X1[],int n,float X2[],float z[],float Wji[][5],float Wkj[][10], int key){

	float X1_bias[20];
	X1_bias[0]=1.0;
	for(int j=1;j<17;j++){
		X1_bias[j]=X1[j-1];
	}
	for(int j=0;j<5;j++){
	    float sum=0;
		for(int i=0;i<17;i++){
			sum=sum+X1_bias[i]*Wji[i][j];
		}
		X2[j]=sigmoid(sum);
	}
	
	float X2_bias[20];
	X2_bias[0]=1.0;
	for(int k=1;k<5;k++){
		X2_bias[k]=X2[k-1];
	}
	
	for(int k=0;k<10;k++){
	    float sum1=0;
	    for(int j=0;j<6;j++){
	        sum1=sum1+X2_bias[j]*Wkj[j][k];  
	    }
	    z[k]=sigmoid(sum1);
	}
	
}

void backward_propagation(float t[], float X1[], float z[], float X2[], float Wkj[][10], float Wji[][5], float D_Wkj[][10], float D_Wji[][5], float eta){
	    
       
	    float delta_output [20]; 
	    for(int k=0; k<10; k++){
	    	delta_output[k] =  ( t[k]-z[k] ) * ( z[k]*(1.0-z[k]) );
	    }		  
	    float X2_bias[20];
	    X2_bias[0]=1.0;
	    for(int k=1;k<5;k++){
		   X2_bias[k]=X2[k-1];
	    }
	    for(int k=0; k<10; k++){
	       for(int j=0; j<6; j++){
		  D_Wkj[j][k] = eta * X2_bias[j] * delta_output[k];
               }
	    }
       
	    float delta_hidden[10]; 
	    for(int j=1; j<6; j++){
	       float sum = 0;
	       for(int r=0; r<10; r++){
		  sum = sum+( delta_output[r] * Wkj[j][r] * ( X2[j-1]*(1.0-X2[j-1]) ) );
	       }
	       delta_hidden[j-1] = sum;
	    }
		
            float X1_bias[20];
	    X1_bias[0]=1.0;
	    int h_index;
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
float RandomNumber(float Min, float Max)
{
    //return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
	return ((float)(rand()%5+1)/100);
}
// calculating norm
float norm_calculation(int nrow, int ncol,float A[][ncol]){
	int i,j;
	float norm;
	float sum=0;
	for(i=0; i<nrow; i++){
		for(j=0; j<ncol; j++){
			sum = sum+ (A[i][j]*A[i][j]);
		}
	}
    norm=sqrt((float)sum);
    return (norm);    
}
int main(){
    //------------------------------------------------- random initialization to weights--------------------------------------------------
     
     float X2[20];
     float z[20];
     float Wji[20][5];
     float Wkj[20][10];
     float D_Wkj_ofNew[20][20];
     float D_Wji_ofNew[20][20];
     float D_Wkj[20][10];
     float D_Wji[20][5];
     srand(time(NULL));
    for(int i=0; i<17; i++){
    	for(int j=0; j<5; j++){
    		Wji[i][j] = RandomNumber(0.01,0.09);
    	 }
    }
    printf("initial Wji weights\n");
    for(int i=0; i<17; i++){
    	for(int j=0; j<5; j++){
    		printf("%f ", Wji[i][j]);
    	}
    	printf("\n");
    }
    
    
    float norm_ofWji;
    norm_ofWji = norm_calculation(17,5,D_Wji_ofNew);
    //printf("%f\n", norm_ofWji);
    for(int j=0; j<6; j++){
        for(int k=0; k<10; k++){
            Wkj[j][k] = RandomNumber(0.01,0.09);
            //D_Wkj_ofNew[j][k] = Wkj[j][k];
        }
    }
 
    printf("initial Wkj weights\n");

    FILE *f = fopen("train1.txt", "r");
		float A[3000][50];
		while(getc(f)!= EOF){
			for(int i=0;i<2216;i++){
				for(int j=0;j<17;j++){
					fscanf(f,"%f",&A[i][j]);
				}
			}
		}

    int epho=0;
	while( epho < 1000){
			for(int i=0; i< 2216; i++){
				float X[20];
				for(int j=1; j<17; j++){
					X[j-1] = A[i][j];
				}

				int class_label = (int)A[i][0];// assigning the label value
				//printf("%d \n", class_label);
				float T[15];
				for(int j=0; j<10; j++){
						T[j]=0.0;
					}	
				T[class_label-1]=1.0;
				forward_propagation(X,5,X2,z,Wji,Wkj,0);
				backward_propagation(T,X,z,X2,Wkj,Wji,D_Wkj,D_Wji,0.001);
			
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
		       }        
        
                       epho++; 
 
    }
    
	
	for(int i=0; i<17; i++){
    	    for(int j=0; j<5; j++){
    		printf("%f ", Wji[i][j]);
    	}
    	   printf("\n");
	 }
	 printf("--------------------\n");
	 for(int j=0; j<6; j++){
                   for(int k=0; k<10; k++){
                      printf("%f ", Wkj[j][k]);
                     }
                    printf("\n");
                  }
        // printf("--------------------\n");
	 FILE *f1= fopen("test.txt", "r");
		float Atest[3000][50];
		while(getc(f1)!= EOF){
			for(int i=0;i<999;i++){
				for(int j=0;j<17;j++){
					fscanf(f1,"%f",&Atest[i][j]);
				}
			}
		}
		
		int label[999];
		float test_out[1000][20];
		for(int i=0;i<999;i++){
		   float Xtest[20];
		   for(int j=1; j<17; j++){
		      Xtest[j-1] = Atest[i][j];
		   }
		   //forward_propagation(Xtest,5,X2,z,Wji,Wkj,1);
		float X1_bias[20];
	        X1_bias[0]=1.0;
	        for(int j=1;j<17;j++){
		    X1_bias[j]=Xtest[j-1];
	        }
	        for(int j=0;j<5;j++){
	           float sum=0;
		for(int l=0;l<17;l++){
		   sum=sum+X1_bias[l]*Wji[l][j];
		}
		   X2[j]=sigmoid(sum);
	       }
	
	      float X2_bias[20];
	      X2_bias[0]=1.0;
	      for(int k=1;k<5;k++){
		 X2_bias[k]=X2[k-1];
	      }
	
	      for(int k=0;k<10;k++){
	         float sum1=0;
	         for(int j=0;j<6;j++){
	            sum1=sum1+X2_bias[j]*Wkj[j][k];  
	         }
	         z[k]=sigmoid(sum1);
              }
		   float max=INT_MIN * 1.0;
		  // printf("----initial MAx:%f\n",max);
		   int index;
		   for(int p=0;p<10;p++){
		    // printf("%f ",z[p]);
		       if(z[p] > max){
		   	 max = z[p];
		   	 index = p;
		   	}
		   }
		   //printf("%d\n",index);
		   label[i]=index+1;
         }
	 float W[17]={4,4,5,7,2,7,7,8,8,6,6,11,1,8,5,11};
	 forward_propagation(W,5,X2,z,Wji,Wkj,1);
	 float max=INT_MIN * 1.0;
	 int index;
	 for(int p=0;p<10;p++){
		      printf("%.2f ",z[p]);
		      //if(p==5)p++;
		       if(z[p] > max){
		   	max = z[p];
         		index = p;
		   }
	}
	printf("%d\n",index+1);	      
	 for(int i=0;i<999;i++){
	    //printf("%d\n  ",label[i] );
	   }
	 printf("\n");   
	//printf("%f\n", avg_norm);
	return 0;
}
