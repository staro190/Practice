# LeNet

### 환경
- Colab CPU
- `TorchVision` 라이브러리 사용
- `torchvision` 라이브러리에서 `MNIST DS` 다운로드 사용
- 기타(`Matplotlib`, `Numpy`) 등 사용

### Data
1) 데이터 변환 함수 정의
   - 입력되는 데이터를 32 x 32 크기로 변환
```
# Define Transform Function
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
```
2) 데이터 다운로드
   - 변환 함수를 전달하여 다운로드
```
# Define Dataset with Tr Func
from torchvision import datasets

path2data = '/content/data'

train_data = datasets.MNIST(path2data, train=True, download=True, transform = data_transform)
val_data = datasets.MNIST(path2data, train=False, download=True, transform = data_transform)
```
3) 
4) ㅇ
5) 
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

### Training
- Optimizer는 `Adam` 사용
- 손실(목적)함수는 `sparse_categorical_crossentropy` 사용(분류 문제, 정수 레이블)
- 성능 확인용 지표 `Accuracy` 이용
- 전체 Epoch : 10
- 배치사이즈 : 64

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(x_train2.shape)
history = model.fit(x_train2, y_train, epochs=10, batch_size = 64, validation_data=(x_val, y_val))
```
![image](https://github.com/staro190/Vision_Practice/assets/16371108/6e361e6d-5efc-47a8-9c4d-650b1b3ae169)


### Validation
- 최적 Epoch를 확인하기 위한 시각화과정
- `Learning Curve`를 `Accuracy`와 `Loss` 2개 관점에서 작성
- 과적합 구간을 확인함

```
# 학습 곡선 준비(Acc)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 학습 곡선 준비(Loss)
loss = history.history['loss']
val_loss = history.history['val_loss']

# 학습 곡선 플롯(Acc)
plt.figure(figsize=(8,8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# 학습 곡선 플롯(Loss)
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```
![image](https://github.com/staro190/Vision_Practice/assets/16371108/ac8a0d29-a893-47f3-827d-998dca142c70)



### Evaluate
- 테스트셋을 이용해서 모델을 평가
- 정확도(`Accuracy`)가 약 90% 임을 확인함
```
model.evaluate(x_test,  y_test, verbose=2)
```
![image](https://github.com/staro190/Vision_Practice/assets/16371108/639641f4-4e11-471b-a66e-62663fcd891a)



### 느낀점
- `Yolo` 등 사용이 간편하게 구성된 모델만 라이브러리를 통해 쉽게 사용했었는데, 이렇게 직접 레이어들을 구성하니 각 레이어의 특징과 역할을 조금 알게되었습니다.
- 모델을 구현하면서 블로그 참조를 많이 하였으나, 기본적으로 모델 그림을 보고 구현하고자 하였습니다.
- 모델 그림을 보고 구현 시, 입출력 Shape와 채널 등을 맞춰주는게 제일 어려웠습니다.
- 특히 `Fashoin Mnist` 데이터셋을 `Alexnet`의 기본 형태인 227×227로 바꿔주는 방법에서 시간이 오래 걸렸습니다.
- 직접 만들고 확인해보니 다른 모델들도 이렇게 구현해보고 싶습니다.
