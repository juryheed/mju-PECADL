import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTConfig
import os
from PIL import Image

class_names = [
    'no_disease_질병없음', 'blepharitis_안검염', 'cataract_백내장', 'conjunctivitis_결막염', 'entropion_안검내반',
    'epiphora_유루증', 'eyelidTumor_안검종양', 'nonUlcerativeCornealDisease_비궤양성각막병',
    'nuclearSclerosis_핵경화', 'pigmentaryKeratitis_색소성각막염', 'ulcerativeCornealDisease_궤양성각막병'
]

symptom_folders = {
    'blepharitis_안검염': ['true', 'false'],
    'cataract_백내장': ['Early', 'false', 'Immature', 'Mature'],
    'conjunctivitis_결막염': ['true', 'false'],
    'entropion_안검내반': ['true', 'false'],
    'epiphora_유루증': ['true', 'false'],
    'eyelidTumor_안검종양': ['true', 'false'],
    'nonUlcerativeCornealDisease_비궤양성각막병': ['down', 'false', 'up'],
    'nuclearSclerosis_핵경화': ['true', 'false'],
    'pigmentaryKeratitis_색소성각막염': ['true', 'false'],
    'ulcerativeCornealDisease_궤양성각막병': ['down', 'false', 'up']
}

symptom_models = {}

# Define a new class with the correct classifier size
class ViTForImageClassificationWithDropout(nn.Module):
    def __init__(self, config):
        super(ViTForImageClassificationWithDropout, self).__init__()
        self.vit = ViTForImageClassification(config)  # Use ViTForImageClassification as backbone
        self.dropout = nn.Dropout(p=0.5)  # Dropout rate
        self.classifier = nn.Linear(self.vit.config.hidden_size, len(class_names))  # Correct classifier size

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        pooled_output = outputs.logits  # Use logits as pooled output
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        logits = self.classifier(pooled_output)
        return logits  # Return logits directly

# Combine all models
class CombinedModel(nn.Module):
    def __init__(self, disease_classify_model, true_classify_model, symptom_models):
        super(CombinedModel, self).__init__()
        self.disease_classify_model = disease_classify_model
        self.true_classify_model = true_classify_model
        self.symptom_models = symptom_models

    def forward(self, images):
        # 질병 예측
        disease_classify_outputs = self.disease_classify_model(images)
        predicted_disease_idx = disease_classify_outputs.argmax(1).item()
        predicted_disease = class_names[predicted_disease_idx + 1]  # '질병 없음' 제외

        # 증상 예측
        if predicted_disease in self.symptom_models:
            symptom_model = self.symptom_models[predicted_disease]
            symptom_outputs = symptom_model(images)
            predicted_symptom_idx = symptom_outputs.argmax(1).item()
            predicted_symptom = symptom_folders[predicted_disease][predicted_symptom_idx]
            return f"{predicted_disease}, {predicted_symptom}"
        else:
            return predicted_disease

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 질병 분류 모델 구성 불러오기
disease_classify_model_name = "google/vit-base-patch16-224"

# 질병 분류 모델 초기화
disease_classify_config = ViTConfig.from_pretrained(disease_classify_model_name)
disease_classify_model = ViTForImageClassificationWithDropout(config=disease_classify_config)

# Load the state dict
disease_classify_model_path = 'C:/Users/dlcks/peteyte/pythonProject/trueClassfication.pth'  # <-- 모델 파일 경로 입력
state_dict = torch.load(disease_classify_model_path, map_location=torch.device('cpu'))['model_state_dict']

# Check if the state_dict contains the classifier weights and biases
classifier_weight_key = 'classifier.weight'
classifier_bias_key = 'classifier.bias'

if classifier_weight_key in state_dict and classifier_bias_key in state_dict:
    # Get the number of output classes in the current model
    new_num_classes = len(class_names)

    # Get the shape of the old classifier weight
    old_classifier_weight_shape = state_dict[classifier_weight_key].shape

    # Create new classifier weight and bias
    new_classifier_weight = state_dict[classifier_weight_key][:new_num_classes, :]
    new_classifier_bias = state_dict[classifier_bias_key][:new_num_classes]

    # Update the state_dict with the new classifier weights and biases
    state_dict[classifier_weight_key] = new_classifier_weight
    state_dict[classifier_bias_key] = new_classifier_bias

    # Load modified state_dict
    disease_classify_model.load_state_dict(state_dict, strict=False)

    print("Modified classifier weights and biases loaded successfully!")
else:
    print("Classifier weights or biases not found in the state_dict.")
disease_classify_model.to(device)

# Load true classify model
true_classify_model_name = "google/vit-base-patch16-224"
true_classify_config = ViTConfig.from_pretrained(true_classify_model_name)
true_classify_config.num_labels = 4  # 증상 있음 / 없음
true_classify_model = ViTForImageClassification(config=true_classify_config)

# 모델 파일 경로와 장치 설정
true_classify_model_path = 'C:/Users/dlcks/peteyte/pythonProject/diseaseClassfication_vit.pth'  # <-- 모델 파일 경로 입력
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the state dict
state_dict = torch.load(true_classify_model_path, map_location=device)

# Get old classifier weight and bias
old_classifier_weight = state_dict['classifier.weight']
old_classifier_bias = state_dict['classifier.bias']

# Ensure the new classifier has the right size
new_classifier_weight = torch.zeros((true_classify_config.num_labels, old_classifier_weight.shape[1]))
new_classifier_bias = torch.zeros((true_classify_config.num_labels))

# Copy the weights and biases from the old classifier to the new one
new_classifier_weight[:old_classifier_weight.shape[0], :] = old_classifier_weight[:true_classify_config.num_labels, :]
new_classifier_bias[:old_classifier_bias.shape[0]] = old_classifier_bias[:true_classify_config.num_labels]

# Replace the classifier weights and biases in the state_dict
state_dict['classifier.weight'] = new_classifier_weight
state_dict['classifier.bias'] = new_classifier_bias

# Load modified state_dict
true_classify_model.load_state_dict(state_dict, strict=False)
true_classify_model.to(device)

print("Modified classifier weights and biases loaded successfully!")

# Combine all models into one
combined_model = CombinedModel(disease_classify_model, true_classify_model, symptom_models)
combined_model.to(device)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    result = combined_model(image)
    return result

# Predict
image_path = 'D:/eye/eyedata/data/original/orgin/disease/disease/crop_D0_0d4f0dab-60a5-11ec-8402-0a7404972c70.jpg'
prediction = predict_image(image_path)
print("이미지 예측 결과:", prediction)
