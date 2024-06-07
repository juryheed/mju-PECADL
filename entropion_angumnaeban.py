import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import warnings

# 특정 경고 억제
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention", category=UserWarning)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 데이터셋 경로
train_dir = 'D:\eye\eyedata\data\Training\label\dog\eye_ball\camaera\entropion_angumnaeban'
val_dir = 'D:\eye\eyedata\data\Validation\dog_val\entropion_angumnaeban_vali'
test_dir = 'D:\eye\eyedata\data\Validation\dog_val\entropion_angumnaeban_vali'

# 데이터셋 경로 존재 여부 확인
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")

# 이미지 전처리 및 데이터 증강
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 생성
train_dataset = ImageFolder(root=train_dir, transform=transform_train)
val_dataset = ImageFolder(root=val_dir, transform=transform_val)
test_dataset = ImageFolder(root=test_dir, transform=transform_val)

# 데이터로더 설정
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 드롭아웃 추가를 위한 ViT 모델 확장
class ViTForImageClassificationWithDropout(ViTForImageClassification):
    def __init__(self, config):
        super(ViTForImageClassificationWithDropout, self).__init__(config)
        self.dropout = nn.Dropout(p=0.5)  # 드롭아웃 비율 설정

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)  # 드롭아웃 적용
        logits = self.classifier(pooled_output)
        return logits

print(f"클래스 수: {len(train_dataset.classes)}")
print(f"클래스 목록: {train_dataset.classes}")


# 모델 초기화
model_name = "google/vit-base-patch16-224"
config = ViTConfig.from_pretrained(model_name)
config.num_labels = len(train_dataset.classes)  # 출력 레이어의 크기 수정
model = ViTForImageClassificationWithDropout(config).to(device)


# 기존 모델 로드
model_path = 'entropion_angumnaeban_vit.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 기존 옵티마이저 상태 로드 (옵션)
optimizer_path = 'optimizer_state_entropion_angumnaeban.pth'
if os.path.exists(optimizer_path):
    optimizer.load_state_dict(torch.load(optimizer_path))
    print(f"Loaded optimizer state from {optimizer_path}")

# 학습 과정에서 손실과 정확도 기록을 위한 변수 초기화
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 기존 기록 로드 (옵션)
record_path = 'training_records_entropion_angumnaeban.pth'
if os.path.exists(record_path):
    records = torch.load(record_path)
    train_losses = records['train_losses']
    val_losses = records['val_losses']
    train_accuracies = records['train_accuracies']
    val_accuracies = records['val_accuracies']
    print("Loaded training records")

# 모델 훈련
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accuracy = correct / total
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 모델 및 옵티마이저 상태 저장
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    # 학습 기록 저장
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }, record_path)

# 학습 과정 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.show()

# 모델 평가 함수
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    return test_loss, test_accuracy

# 테스트 데이터로 모델 평가
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
