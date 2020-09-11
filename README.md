# pothole

개요
-----------
자동차의 달려있는 센서 데이터 (x, y, z) 를 이용하여 도로의 깨진 부분을 찾는 pothole을 탐지
이전 실험에서 SVM을 이용해 자동차의 시계열 센터 데이터를 통해 pothole을 검출하였다.

현 실험에서는 CNN 을 이용하여 pothole을 검출하였다.

![image](https://user-images.githubusercontent.com/65576979/92912168-7ed4be80-f464-11ea-8f7a-384df6746a6e.png)

SVM의 DTW를 이용해 pothole을 검출

![image](https://user-images.githubusercontent.com/65576979/92912291-99a73300-f464-11ea-9754-6c0722bdac46.png)

CNN에서는 SVM처럼 DTW를 사용할 수 없기 때문에 데이터를 특정 크기(윈도우) 만큼 슬라이스


데이터
-------------
데이터는 차량의 시간,센서 데이터(x, y, z), 위도, 경도으로 되어있으며
이 시계열 데이터를 window 크기와 step 크기에 따라 슬라이스하여 학습을 실시하였다.

네트워크
----------------
![image](https://user-images.githubusercontent.com/65576979/92912409-bb081f00-f464-11ea-8b2f-9fd7b7191a3e.png)
pothole 검출에 사용한 CNN

실험결과
-------------
가로축은 SVM 을 시작해 각 window 크기를 나타내는 W, step 크기를 나타내는 S 가 있다.
비교 부분은 Accuracy, FalseNagative, TruePositive, F1 score 로 나온다.

![image](https://user-images.githubusercontent.com/65576979/92886093-5641ca00-f44e-11ea-8706-c96ad3332164.png)
