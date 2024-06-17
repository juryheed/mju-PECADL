import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
import os
import warnings

warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention", category=UserWarning)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

# 클래스 이름 및 증상 폴더 이름
class_names = [
    'blepharitis_안검염', 'cataract_백내장', 'conjunctivitis_결막염', 'entropion_안검내반',
    'epiphora_유루증', 'eyelidTumor_안검종양', 'nonUlcerativeCornealDisease_비궤양성각막병',
    'nuclearSclerosis_핵경화', 'pigmentaryKeratitis_색소성각막염', 'ulcerativeCornealDisease_궤양성각막병'
]

symptom_folders = {
    'blepharitis_안검염': ['유', '무'],
    'cataract_백내장': ['초기', '무', '미성숙', '성숙'],
    'conjunctivitis_결막염': ['유', '무'],
    'entropion_안검내반': ['유', '무'],
    'epiphora_유루증': ['유', '무'],
    'eyelidTumor_안검종양': ['유', '무'],
    'nonUlcerativeCornealDisease_비궤양성각막병': ['상', '무', '하'],
    'nuclearSclerosis_핵경화': ['유', '무'],
    'pigmentaryKeratitis_색소성각막염': ['유', '무'],
    'ulcerativeCornealDisease_궤양성각막병': ['상', '무', '하']
}

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 로드
model_name = "google/vit-base-patch16-224"
config = ViTConfig.from_pretrained(model_name)
config.num_labels = sum(len(v) for v in symptom_folders.values())  # 총 클래스 수 설정
model = ViTForImageClassification(config=config)

# 모델 경로 설정 및 로드 (옵션)
model_dir = 'C:/Users\dlcks\peteyte\pythonProject'  # 여기에 모델 파일이 있는 폴더 경로를 설정하세요
model_paths = {
    'blepharitis_안검염': os.path.join(model_dir, 'blepharitis_angumyeom_vit.pth'),
    'cataract_백내장': os.path.join(model_dir, 'cataract_backnaejang_vit.pth'),
    'conjunctivitis_결막염': os.path.join(model_dir, 'conjunctivitis_gyulmak.pth'),
    'entropion_안검내반': os.path.join(model_dir, 'entropion_angumnaeban_vit.pth'),
    'epiphora_유루증': os.path.join(model_dir, 'epiphora_youroozeong_vit.pth'),
    'eyelidTumor_안검종양': os.path.join(model_dir, 'eyelidTumor_angumjongyang_vit.pth'),
    'nonUlcerativeCornealDisease_비궤양성각막병': os.path.join(model_dir, 'nonUlcerativeCornealDisease_begaeyangsung_vit.pth'),
    'nuclearSclerosis_핵경화': os.path.join(model_dir, 'nuclearSclerosis_hackcgyeonghwa_vit.pth'),
    'pigmentaryKeratitis_색소성각막염': os.path.join(model_dir, 'pigmentaryKeratitis_color_vit.pth'),
    'ulcerativeCornealDisease_궤양성각막병': os.path.join(model_dir, 'ulcerativeCornealDisease_gaeyangsung_vit.pth')
}

for disease, path in model_paths.items():
    if os.path.exists(path):
        print(f"모델 파일 {path}을(를) 로드합니다.")
        state_dict = torch.load(path)

        # 분류기 레이어의 가중치를 제외한 나머지 가중치를 로드
        state_dict.pop('classifier.weight', None)
        state_dict.pop('classifier.bias', None)

        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"모델 파일 {path}을(를) 찾을 수 없습니다. 기본 모델을 사용합니다.")

# 모델을 GPU로 이동
model.to(device)

# 데이터셋 경로 설정 및 생성
data_dir = 'D:\\eye\\eyedata\\data\\original\\orgin\\disease'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 증상 예측 횟수를 저장할 딕셔너리 초기화
prediction_counts = {disease: {symptom: 0 for symptom in symptoms} for disease, symptoms in symptom_folders.items()}

# 예측 및 결과 출력
for idx, (images, _) in enumerate(loader):
    images = images.to(device)
    outputs = model(images)
    predicted = outputs.logits.argmax(1)  # 예측 결과

    print(f"사진 {idx + 1}의 예측 결과: {predicted.item()}")

    # 예측된 클래스 인덱스를 각 질병에 대해 매핑
    class_offset = 0
    prediction_map = {}
    for class_name, symptoms in symptom_folders.items():
        if predicted.item() >= class_offset and predicted.item() < class_offset + len(symptoms):
            predicted_symptom = symptoms[predicted.item() - class_offset]
            prediction_map[class_name] = predicted_symptom
            prediction_counts[class_name][predicted_symptom] += 1  # 예측 횟수 증가
        else:
            prediction_map[class_name] = '무'
        class_offset += len(symptoms)

    # 각 질병에 대해 예측 결과 출력
    for i, class_name in enumerate(class_names):
        print(f"{class_name} 증상= {prediction_map[class_name]}")

    print()  # 각 사진의 결과를 분리하기 위해 개행

# 최종 예측 횟수 출력
print("증상 예측 횟수:")
for class_name, counts in prediction_counts.items():
    print(f"{class_name}:")
    for symptom, count in counts.items():
        print(f"  {symptom}: {count}")
