import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt

# GPU 비활성화 설정
def disable_gpu():
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU has been disabled.")
    except Exception as e:
        print(f"Error disabling GPU: {e}")

disable_gpu()

# 데이터 경로 설정
base_dir = 'D:/download/eye_data/datas/Validation/images'

# 이미지 크기 설정
img_height, img_width = 256, 400
batch_size = 32
epochs = 50  # 이어서 학습할 에포크 수
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
print('1')
# %%
# 모델 불러오기
loaded_model = tf.keras.models.load_model('dog_classifier_final_continued2')

# 모델 컴파일 (필요한 경우, 불러온 모델에 대해 다시 컴파일할 수 있습니다)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 체크포인트 및 얼리 스토핑 콜백 설정
checkpoint = ModelCheckpoint(
    'dog_classifier_continued',  # 확장자 없이 파일 이름만 지정
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    save_format='tf'  # TensorFlow 형식으로 저장
)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print('2')
# %%
# 모델 이어서 학습
history = loaded_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, TqdmCallback(verbose=1)]
)
print('3')
# %%
# 테스트 데이터로 모델 평가
test_loss, test_acc = loaded_model.evaluate(test_generator, callbacks=[TqdmCallback(verbose=1)])
print(f"Test Accuracy: {test_acc}")
print('4')
# %%
# 학습 과정 시각화
plt.figure(figsize=(12, 4))

# 정확도 시각화
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# 손실 시각화
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
print('5')
# %%
# 모델 저장
loaded_model.save('dog_classifier_final_continued2', save_format='tf')
print('6')
