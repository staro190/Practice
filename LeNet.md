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
3) 데이터 로더 정의
   - 배치 사이즈로 데이터를 불러오는 로더 정의
```
# Define Loader with Dataset
from torch.utils.data import DataLoader

train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
val_dl = DataLoader(val_data, batch_size= 32)
```

### Model

1) 모델 클래스 정의
   - LeNet_5 구조 사용
   - `torch.nn` 라이브러리의 `Module` 클래스 상속
```
from torch import nn
import torch.nn.functional as F
import torch
class LeNet_5(nn.Module):
  def __init__(self):
    super(LeNet_5, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
    self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

    self.fc1 = nn.Linear(120, 84)
    self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.tanh(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)

    x = F.tanh(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)

    x = F.tanh(self.conv3(x))
    x = x.view(-1, 120)

    x = F.tanh(self.fc1(x))
    x = self.fc2(x)

    return F.softmax(x, dim=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = LeNet_5().to(device)
print(model)

from torchsummary import summary
summary(model, input_size=(1,32,32))

```

### Training

1) 학습 옵션 설정
   - Optimizer : `Adam`
   - Loss Function : `CrossEntropyLoss`, Batch에 대하여 `sum`
   - learning rate schedule : `CosineAnnealingLR` , `Cosine` 함수 형태로 `Learning Rate`가 변화

```
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-05)
```
2) 학습에 사용될 함수 정의
   - get_lr : 최적 파라미터 딕셔너리에서 lr 전달
   - metrics_batch : 예측과 정답을 비교하여 정답(TP) 개수를 정수로 반환
   - loss_batch : 손실함수 적용, 기울기 변수 초기화 및 역전파(1step)
   - loss_epoch : 1 epoch 적용
```
import torch

def get_lr(opt):
  for param_group in opt.param_groups:
    return param_group['lr']

def metrics_batch(output, target):
  # softmax -> argmax , prediction number(0~9)
  pred = output.argmax(dim=1, keepdim=True)
  # calculate number of correct
  corrects = pred.eq(target.view_as(pred)).sum().item()
  return corrects

def loss_batch(loss_func, output, target, opt=None):
  loss = loss_func(output, target)
  metric_b = metrics_batch(output, target)
  # opt initailize, 
  if opt is not None:
    # pre step's backward gradient initailize
    opt.zero_grad()
    # loss back propagation
    loss.backward()
    # add calculated grad on parameter
    opt.step()
  return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
  running_loss = 0.0
  running_metric = 0.0
  len_data = len(dataset_dl.dataset)

  # epoch running(batch)
  for xb, yb in dataset_dl:
    # batch size data load to device
    xb = xb.type(torch.float).to(device)
    yb = yb.to(device)
    # batch data input to model
    output = model(xb)
    loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
    running_loss += loss_b

    if metric_b is not None:
      running_metric += metric_b

    # for check
    if sanity_check is True:
      break

  # Average Loss of 1 epoch
  loss = running_loss/float(len_data)
  # Average Accuract of 1 epoch
  metric = running_metric/float(len_data)
  return loss, metric

```
3) 학습 함수
   - 전체 `epoch` 만큼 진행
   - `learning rate` 관리, 모델 `train`/`eval`
   - Val Set에 대하여 계산 및 최적 모델 상태 탐색 / 저장
```
import copy

def train_val(model, params):
  num_epochs = params['num_epochs']
  loss_func = params['loss_func']
  opt = params['optimizer']
  train_dl = params['train_dl']
  val_dl = params['val_dl']
  sanity_check = params['sanity_check']
  lr_scheduler = params['lr_sheduler']
  path2weights = params['path2weights']

  loss_history = {
      'train' : [],
      'val': [],
  }

  metric_history = {
      'train' : [],
      'val': [],
  }

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = float('inf')

  for epoch in range(num_epochs):
    current_lr = get_lr(opt)
    print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))
    model.train()
    train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)

    loss_history['train'].append(train_loss)
    metric_history['train'].append(train_metric)

    model.eval()
    # grad update off
    with torch.no_grad():
      # just cal of validation set / model parameter's are freezed / one epoch cal
      val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
      loss_history['val'].append(val_loss)
      metric_history['val'].append(val_metric)

    # find Best State
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
      #torch.save(model.state_dict(), path2weights)
      print('Copied best model weights')

    # update lr schedule
    lr_scheduler.step()

    print('train loss: %.6f, dev loss: %.6f, accuracy: %.2f' %(train_loss, val_loss, 100*val_metric))
    print('-'*10)

  # return model's best state / history
  model.load_state_dict(best_model_wts)
  return model, loss_history, metric_history
```
4) 학습
   - 하이퍼파라미터 지정 후 학습함수 실행
```

params_train = {}
params_train['num_epochs'] = 100
params_train['loss_func'] = loss_func
params_train['optimizer'] = opt
params_train['train_dl'] = train_dl
params_train['val_dl'] = val_dl
params_train['sanity_check'] = True # Check
params_train['lr_sheduler'] = lr_scheduler
params_train['path2weights'] = './'

model,loss_hist,metric_hist=train_val(model ,params_train)
```

### Validation
- 최적 Epoch를 확인하기 위한 시각화과정
- `Learning Curve`를 `Accuracy`와 `Loss` 2개 관점에서 작성
- 과적합 구간을 확인함

```
#LOSS
num_epochs=params_train["num_epochs"]

plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
```
```
# ACC
# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
```

