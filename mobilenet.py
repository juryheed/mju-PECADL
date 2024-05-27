import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm.keras import TqdmCallback

# 데이터 경로 설정
base_dir = 'D:/download/eye_data/datas/Validation/images'

# 이미지 크기 설정
img_height, img_width = 256, 400
batch_size = 32
epochs = 10
learning_rate = 0.001

# 데이터 증강 및 제너레이터 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 검증 데이터로 20% 분할
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 제너레이터 생성
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# 모델 구성
base_model = MobileNet(weights=None, include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 베이스 모델의 모든 층을 고정 (학습되지 않도록 설정)
for layer in base_model.layers:
    layer.trainable = True  # 처음부터 학습할 것이므로 모든 레이어를 학습 가능하도록 설정

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 체크포인트 콜백 설정
checkpoint = ModelCheckpoint('mobilenet_eye_classifier.h5', monitor='val_loss', save_best_only=True, mode='min')

# 모델 학습
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, TqdmCallback()]
)

# 테스트 데이터로 모델 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# 모델 저장
model.save('mobilenet_eye_classifier_final.h5')
