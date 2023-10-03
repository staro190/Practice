# Alexnet

### Library
- `TensorFlow` 라이브러리 중 `Keras` 사용
- `Sklearn` 라이브러리에서 `Fashion Mnist` 다운로드 사용
- 기타(`Matplotlib`, `Numpy`) 등 사용

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
```

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
- 파라미터 개수는 위 그림과 동일, 패딩은 모두 1칸(`same` 옵션)

```
# 모델 구성
model = keras.Sequential()

# 그레이스케일 이미지에 맞춰 Train 데이터 Shape 변경
x_train2 = np.reshape(x_train, (60000, 28, 28, 1))

# Conv 1
model.add(layers.experimental.preprocessing.Resizing(227, 227, interpolation='bilinear',input_shape=x_train2.shape[1:]))
model.add(layers.Conv2D(96,(11,11), strides=(4,4), activation='relu',padding='same'))
model.add(layers.MaxPool2D((3,3), strides=2))
model.add(layers.BatchNormalization())

# Conv 2
model.add(layers.Conv2D(256,(5,5), strides=1, activation='relu',padding='same'))
model.add(layers.MaxPool2D((2,2), strides=2))
model.add(layers.BatchNormalization())

# Conv 3
model.add(layers.Conv2D(384,(3,3), strides=1, activation='relu',padding='same'))
model.add(layers.Conv2D(384,(3,3), strides=1, activation='relu',padding='same'))
model.add(layers.Conv2D(384,(3,3), strides=1, activation='relu',padding='same'))
model.add(layers.MaxPool2D((3,3), strides=2))
model.add(layers.BatchNormalization())

# FC
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# 알렉스넷 구성 확인
print(model.summary())
```
![image](https://github.com/staro190/Vision_Practice/assets/16371108/6f680df6-9066-411d-9edd-d2c7243eca12)

