# Alexnet

### Data
- 사이킷런 라이브러리에서 제공하는 `Fashion Mnist` 데이터셋 이용
- 60,000 개의 훈련 데이터와 10,000개의 테스트셋으로 구성
- 테스트셋을 2개로 분리하여 검증셋과 테스트셋 구성
- 데이터를 안정적인 비율로 구성하기 위하여 8:1:1로 구성

```
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1, shuffle=False)
```

### Model
![image](https://github.com/staro190/Vision_Practice/assets/16371108/95107f3a-e32a-4820-8fd2-aba7ae97ff3a)
- AlexNet 기본 구조 사용
- `Fashion Mnist`는 (28 × 28) 크기이므로 (227 × 227) 크기로 데이터 확장
- `Fashion Mnist`는 그레이스케일 이미지이므로 위 그림에서 채널은 1로 봐야함
- Conv, Pool, Norm 으로 구성된 레이어 2개와 Conv × 3, Pool, Norm 1개, FC 3개로 구성
- 파라미터 개수는 위 그림과 동일
