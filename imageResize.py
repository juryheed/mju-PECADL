import cv2
import os

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseA'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseA'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_folder_path, filename)
        output_image_path = os.path.join(output_folder_path, filename)

        # 이미지 읽기
        image = cv2.imread(input_image_path)

        if image is not None:
            # 이미지 리사이즈 (CPU를 사용한 리사이즈)
            resized_image = cv2.resize(image, (400, 256))

            # 리사이즈된 이미지 저장
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized and saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")

print("All images have been resized.")

# ----------------------------------------

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseB'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseB'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_folder_path, filename)
        output_image_path = os.path.join(output_folder_path, filename)

        # 이미지 읽기
        image = cv2.imread(input_image_path)

        if image is not None:
            # 이미지 리사이즈 (CPU를 사용한 리사이즈)
            resized_image = cv2.resize(image, (400, 256))

            # 리사이즈된 이미지 저장
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized and saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")

print("All images have been resized.")

# ----------------------------------------

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseC'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseC'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_folder_path, filename)
        output_image_path = os.path.join(output_folder_path, filename)

        # 이미지 읽기
        image = cv2.imread(input_image_path)

        if image is not None:
            # 이미지 리사이즈 (CPU를 사용한 리사이즈)
            resized_image = cv2.resize(image, (400, 256))

            # 리사이즈된 이미지 저장
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized and saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")

print("All images have been resized.")

# ----------------------------------------

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseD'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseD'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_folder_path, filename)
        output_image_path = os.path.join(output_folder_path, filename)

        # 이미지 읽기
        image = cv2.imread(input_image_path)

        if image is not None:
            # 이미지 리사이즈 (CPU를 사용한 리사이즈)
            resized_image = cv2.resize(image, (400, 256))

            # 리사이즈된 이미지 저장
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized and saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")

print("All images have been resized.")

# ----------------------------------------

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseE'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseE'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_folder_path, filename)
        output_image_path = os.path.join(output_folder_path, filename)

        # 이미지 읽기
        image = cv2.imread(input_image_path)

        if image is not None:
            # 이미지 리사이즈 (CPU를 사용한 리사이즈)
            resized_image = cv2.resize(image, (400, 256))

            # 리사이즈된 이미지 저장
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized and saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")

print("All images have been resized.")

# ----------------------------------------

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseF'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/images/diseaseF'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_folder_path, filename)
        output_image_path = os.path.join(output_folder_path, filename)

        # 이미지 읽기
        image = cv2.imread(input_image_path)

        if image is not None:
            # 이미지 리사이즈 (CPU를 사용한 리사이즈)
            resized_image = cv2.resize(image, (400, 256))

            # 리사이즈된 이미지 저장
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized and saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")

print("All images have been resized.")



# 이하 중분류 만큼 복붙
