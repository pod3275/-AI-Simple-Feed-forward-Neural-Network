# Simple-Feed-forward-Neural-Network
 - 2017년 1학기 성균관대학교 이지형 교수님 인공지능 수업 3번째 과제
 - 간단한 feed-forward neural network 코드를 이용하여 특정 데이터를 training 및 test를 실행하는 과제
 - 2017 1st semester Sungkyunkwan University professor Jee Hyong Lee's Artificial Intelligence class, 3rd assignment
 - Train and test specific data using simple feed-forward neural network code

## 1. Problem
 - Simple feed-forward neural network의 구조 (코드 내 파라미터)
   - NUM_INPUT : number of inputs
   - NUM_HIDDEN : number of nodes in the hidden layer
   - NUM_OUTPUT : number of outputs
   - NUM_TRAINING_DATA : number of data for training
   - NUM_TEST_DATA : number of data for test
   - MAX_EPOCH : number of iterations for learning
   - LEARNING_RATE : learning rate (Eta)

-----------------------------------------------
### (1) Run example in sample code
 - 주어진 training dataset을 학습하는 neural network를 설계하시오.  
 학습 후의 final weight을 포함한 neural network의 구조를 그리고, 주어진 test dataset에 대한 결과를 나타내시오.
 
#### Example 1
 - XOR 학습 neural network
 - Training data  
   > (x1, x2, y) : (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 0.0)
 - Test data  
   > (x1, x2) : (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)
   
 - 학습 파라미터 설정   
   `#define NUM_INPUT	2`  
   `#define	NUM_HIDDEN	2`  
   `#define	NUM_OUTPUT	1`     
   `#define	NUM_TRAINING_DATA	4`   
   `#define	NUM_TEST_DATA	4`  
   `#define	MAX_EPOCH	100000`  
   `#define	LEARNING_RATE	0.5`
      
#### Example 2
 - XOR과 OR 학습 neural network
 - Training data  
   > (x1, x2, y1, y2) : (1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)
 - Test data  
   > (x1, x2) : (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)
   
 - 학습 파라미터 설정   
   `#define NUM_INPUT	2`  
   `#define	NUM_HIDDEN	2`  
   `#define	NUM_OUTPUT	2`     
   `#define	NUM_TRAINING_DATA	4`   
   `#define	NUM_TEST_DATA	4`  
   `#define	MAX_EPOCH	100000`  
   `#define	LEARNING_RATE	0.5`
      
#### Example 3
 - y = 4x*(1-x) 학습 neural network
 - Training data  
   > (x, y) : (0.0, 0.00), (0.1, 0.36), (0.2, 0.64), (0.3, 0.84), (0.4, 0.96), (0.5, 1.0)  
   (0.6, 0.96), (0.7, 0.84), (0.8, 0.64), (0.1, 0.36), (1.0, 0.00)
 - Test data  
   > (x) : (0.0), (0.02), (0.04), ..., (0.98), (1.0)
   
 - 학습 파라미터 설정   
   `#define NUM_INPUT	1`  
   `#define	NUM_HIDDEN	5`  
   `#define	NUM_OUTPUT	1`     
   `#define	NUM_TRAINING_DATA	11`   
   `#define	NUM_TEST_DATA	51`  
   `#define	MAX_EPOCH	100000`  
   `#define	LEARNING_RATE	0.5`
   
-----------------------------------------------
### (2) Training with noisy data
 - 다음은 4x*(1-x)를 학습하는 neural network의 training dataset이다. 하지만 약간의 error가 섞여 있다.
 
   ![image](https://user-images.githubusercontent.com/26705935/40634536-742b4c96-6330-11e8-87e7-e99ef67201c7.png)
 
 - 위의 주어진 training dataset을 학습하는, 다음과 같은 조건의 neural network를 만드시오.
   - number of iteration = 5,000,000
   - number of nodes in hidden layer = 10
   - learning rate = 0.5
 
 - 학습 중에 training error와 test error를 계산하고 출력함으로써 다음 표를 작성시오.
 
   ![image](https://user-images.githubusercontent.com/26705935/40634655-280daf06-6331-11e8-938b-97a7fc499e64.png)
   
   - test dataset : x = 0.00, 0.02, ..., 0.98, 1.00
   - error function : E = (o(x)-f(x))^2 / 2
   
-----------------------------------------------
### (3) Detect digits neural network
 - 다음과 같은 문자를 인식하는 neural network를 만드시오.
 
    ![image](https://user-images.githubusercontent.com/26705935/40634736-8860c0c8-6331-11e8-8ad5-35d0cd8f4720.png)

   - number of input = 15
   - number of output = 10
   - training data
      > (x1, x2, ..., x14, x15,  y1, y2, ..., y10) :  
       (1,1,1,1,0,1,1,0,1,1,0,1,1,1,1, 1,0,0,0,0,0,0,0,0,0)  
       (1,1,0,0,1,0,0,1,0,0,1,0,0,1,0, 0,1,0,0,0,0,0,0,0,0)  
       (1,1,1,0,0,1,1,1,1,1,0,0,1,1,1, 0,0,1,0,0,0,0,0,0,0)  
       (1,1,1,0,0,1,1,1,1,0,0,1,1,1,1, 0,0,0,1,0,0,0,0,0,0)  
       (1,0,1,1,0,1,1,1,1,0,0,1,0,0,1, 0,0,0,0,1,0,0,0,0,0)  
       (1,1,1,1,0,0,1,1,1,0,0,1,1,1,1, 0,0,0,0,0,1,0,0,0,0)  
       (1,1,1,1,0,0,1,1,1,1,0,1,1,1,1, 0,0,0,0,0,0,1,0,0,0)  
       (1,1,1,0,0,1,0,0,1,0,0,1,0,0,1, 0,0,0,0,0,0,0,1,0,0)  
       (1,1,1,1,0,1,1,1,1,1,0,1,1,1,1, 0,0,0,0,0,0,0,0,1,0)  
       (1,1,1,1,0,1,1,1,1,0,0,1,1,1,1, 0,0,0,0,0,0,0,0,0,1)  
 
 - 위와 같이 학습한 neural network에 대해 다음과 같은 test set의 결과를 출력하시오.
 
    ![image](https://user-images.githubusercontent.com/26705935/40634869-3e17880c-6332-11e8-8663-103374c00c34.png)

## 2. Environment
 - language : C++
 - IDE : Microsoft Visual studio 2017
 
## 3. Result
### (1) Run example in sample code
 - final weight을 알기 위해 main 함수 끝에 PrintWeight 함수를 이용한  
 
   `printf("\nWeight : ");`     
	  `PrintWeight(weight_kj, weight_ji, bias_k, bias_j);`  
   
   라는 코드를 이용하여 final weight을 출력하도록 하였다.
   
#### Example 1
 - 실행 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635300-68e050a8-6334-11e8-9b5a-473154a2a5d4.png)
   
   - weight 출력 값의 순서는 weight_ji[0][0], weight_ji[0][1], weight_ji[1][0], weight_ji[1][1], bias_j[0], bias_j[1], wieght_kj[0][0], wieght_kj[0][1], bias_k[0] 이다. 각 node는 0번째를 시작으로 count하여 저장한다. weight는 소수점 셋째자리에서 반올림하였다.
   
 - 생성 및 학습된 neural network의 구조
 
   ![image](https://user-images.githubusercontent.com/26705935/40635369-a1a8b632-6334-11e8-9964-bf4bf584ffbc.png)  
   
 - test datset에 대한 output 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635430-ef889c46-6334-11e8-9648-32fc609eb42d.png)
   
     - XOR 연산의 output은 두 input의 값이 다르면 1, 같으면 0이다. 실행 결과와 target을 비교하였을 때, learning을 통해 target과 매우 근접한 neural network를 build하였다고 볼 수 있다.

#### Example 2
 - 실행 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635590-b1b669ba-6335-11e8-8967-b63266ab28c2.png)
   
   - 각 weight 출력 값의 순서는 weight_ji[0][0], weight_ji[0][1], weight_ji[1][0], weight_ji[1][1], bias_j[0], bias_j[1], wieght_kj[0][0], wieght_kj[0][1], wieght_kj[1][0], wieght_kj[1][1], bias_k[0], bias_k[1] 이다. 각 node는 0번째를 시작으로 count하여 저장한다. weight는 소수점 셋째자리에서 반올림하였다.
   
 - 생성 및 학습된 neural network의 구조
 
   ![image](https://user-images.githubusercontent.com/26705935/40635604-cc6a3890-6335-11e8-9bcc-65802636ec36.png)  
   
 - test datset에 대한 output 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635611-d9ba336a-6335-11e8-973b-0bfdde89a35b.png)
   
     - XOR 연산의 output은 두 input의 값이 다르면 1, 같으면 0이고, OR 연산의 output은 input중 1이 1개 이상 있으면 1, 모두 0이면 0이다. 이것은 표에 target1, target2 항목으로 정리되어 있다. 실행 결과와 target을 비교하였을 때, learning을 통해 target과 근접한 neural network를 build하였다고 볼 수 있다.
     
#### Example 3
 - test datset에 대한 output 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635680-2fb3b7c8-6336-11e8-9ed0-4a334c282953.png)
   
   ![image](https://user-images.githubusercontent.com/26705935/40635709-4b05abb2-6336-11e8-973c-dd73f0d5a302.png)
   
   - 순서대로 test input, NN에 의한 output 및 target 값을 나타낸다. Output과 target을 비교하였을 때, learning을 통해 y = 4x(1-x) 함수와 매우 근사하게 구현하였다고 볼 수 있다.
   
 - Final weight 출력 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635738-7540d190-6336-11e8-89f1-c0eaf2e4f77e.png)
   
   - weight 출력 값의 순서는 weight_ji[0][0], weight_ji[1][0], weight_ji[2][0], weight_ji[3][0], weight_ji[4][0], bias_j[0], bias_j[1], bias_j[2], bias_j[3], bias_j[4], wieght_kj[0][0], wieght_kj[0][1], wieght_kj[0][2], wieght_kj[0][3], wieght_kj[0][4], bias_k[0] 이다. 여기서 각 node는 0번째를 시작으로 count하여 저장한다. weight는 소수점 셋째자리에서 반올림하였다.
   
 - 생성 및 학습된 neural network의 구조
 
   ![image](https://user-images.githubusercontent.com/26705935/40635756-8baeb686-6336-11e8-9cba-2a92d9bb5992.png)  
 
-----------------------------------------------
### (2) Training with noisy data
 - 실행 결과
 
   ![image](https://user-images.githubusercontent.com/26705935/40635865-290b7c7a-6337-11e8-980a-d5b688bb8c36.png)
   
     - 순서대로 iteration 수, training error, test error를 나타낸다.
     - training error는 training error = {(training_point에 대한)output – training_target}^2 / 2 로 계산하였다. 
     - testing error를 계산할 때에는 buff = 4*test_point*(1-test_point) 를 따로 저장하였고 이를 통해 testing error = {(test_point에 대한)output – buff}^2 / 2 로 계산하였다.
     
 - 실행 결과 표
 
   ![image](https://user-images.githubusercontent.com/26705935/40635937-9b88df9a-6337-11e8-994f-425242c9c90b.png)
   
     - Learning을 반복할수록 training error는 꾸준히 감소하였으나 testing error는 0회부터 500000회까지 감소하다가 그 이후부터 1000000회까지 반복하는 과정에서 증가하는 경향을 보였다. 이는 training data set의 training target과 실제 값 사이의 오차, 즉 error에 의해 발생한 것이라고 분석된다. 특히 hidden layer의 node수가 10으로 Example 1,2,3에 비해 많아짐에 따라 overfit이 발생한 것이라고 예상한다. 따라서 learning을 반복할수록 neural network는 원래의 함수(y = 4x(1-x)) 와는 다른 형태의 함수의 기능을 갖추게 되고, output이 원래의 함수 값과 점점 멀어져 testing error가 커진다고 볼 수 있다. 이에 반해 output은 training data의 target 값과는 계속하여 가까워지고 training error는 계속하여 작아진다.
 
-----------------------------------------------
### (3) Detect digits neural network
 - 출력 형식
   - Training 시작 시 iteration 횟수 출력.
   - 결과는 output의 범위에 따라 0 또는 1을 저장. 또한 output[k] =1에 해당하는 값 출력.
   - 작성 코드
   
   ![image](https://user-images.githubusercontent.com/26705935/40694811-20599c16-63f9-11e8-812d-4075fa6bd557.png)

 - 실행 결과
   - iteration 횟수(MAX_EPOCH)를 달리하여 결과를 출력하였다.
   - MAX_EPOCH = 1000
   
   ![image](https://user-images.githubusercontent.com/26705935/40694880-91d5c72a-63f9-11e8-9af4-10f663b26465.png)
   
   - MAX_EPOCH = 10000
   
   ![image](https://user-images.githubusercontent.com/26705935/40694887-9c9704c6-63f9-11e8-8bc2-4c171a72016f.png)
   
   - MAX_EPOCH = 50000
   
   ![image](https://user-images.githubusercontent.com/26705935/40694895-a2bd374e-63f9-11e8-9c74-fa16fe84300f.png)
   
   - MAX_EPOCH = 100000
   
   ![image](https://user-images.githubusercontent.com/26705935/40694901-a87a216a-63f9-11e8-9caa-bc1606c339ad.png)
   
   - MAX_EPOCH = 500000
   
   ![image](https://user-images.githubusercontent.com/26705935/40694910-b0d719b2-63f9-11e8-80a6-bb554b1d2563.png)
   
 - 분석
   - 실행 결과 iteration 50000회를 기점으로 성능이 변화한다. 1000회부터 50000회 iteration까지 정확도가 증가하다가 50000회 이상 iteration에서의 정확도는 크게 증가하지 않고, 오히려 감소하기도 하였다. 이는 iteration 횟수가 높을수록 모델의 성능이 좋아지는 것은 아니라는 것임을 나타낸다.

   - 모델의 정확도를 증가시키기 위해서는 가장 효율적인 iteration 횟수를 정하는 것이 중요하다. 또한 hidden layer의 node수, layer의 수 등 모델의 하이퍼파라미터는 성능을 결정하는 중요한 요소이기 때문에 가장 효율적인 하이퍼파라미터를 찾는 것도 중요하다. 

   - 매 실행마다 initial weight이 다르기 때문에 결과가 달라지고, 매 실행 마다 더 정확해지기도 또는 부정확해지기도 하였다. 따라서 initial weight을 성능이 좋을 것이라고 어느 정도 예측한 값으로 설정한다면 매 실행마다 결과가 크게 달라지지 않을 것으로 예상된다.
      
## 4. Future work
 - C++을 이용하여 간단한 Neural Network를 구성해보았다. C++ 언어가 아닌 python의 tensorflow등 다양한 딥러닝 라이브러리를 이용하여 많은 neural network를 구현할 예정이다.
