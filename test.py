import time
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace# 이미지 저장 경로
img1_path = './data/f1.jpg'# 이미지 읽기
img1 = cv2.imread(img1_path)# 이미지 확인
img2_path = 'data/compare_images/d1.jpg'
img2 = cv2.imread(img2_path)
# plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

# 얼굴 검출 모델 목록 (원하는 모델 선택 사용)
# detection_models = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
# # 얼굴 검출 및 정렬 진행
# # face = DeepFace.detectFace(img_path=img,
# #                            detector_backend='retinaface')
# face = DeepFace.extract_faces(img_path=img1, detector_backend='retinaface')[0]['face']
# face_uint8 = (face * 255).astype('uint8')
# # 얼굴 영역 확인
# plt.show()
#
# # 얼굴 표현 모델 목록 (원하는 모델 선택 사용)
# embedding_models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']
# # 얼굴 표현(임베딩)
# embedding = DeepFace.represent(img_path=img1,
#                                detector_backend='retinaface',
#                                model_name='ArcFace')
#
# # 벡터 크기 확인
# print(len(embedding))
# # 벡터값 확인
# print(embedding)

result = DeepFace.verify(img1_path=img1_path,
                         img2_path=img2_path,
                         detector_backend='retinaface',
                         model_name='ArcFace')
# 결과 확인
distance = result['distance']
threshold = result['threshold']
verified = result['verified']
if verified:
    verified_str = 'Same'
    distance_str =  '(%.2f <= %.2f)' % (distance, threshold)
else:
    verified_str = 'Different'
    distance_str =  '(%.2f > %.2f)' % (distance, threshold)# 결과 시각화

fig = plt.figure(figsize=(10, 5))  # 크기를 설정 (선택 사항)
rows, cols = 1, 2

# 첫 번째 이미지 표시
ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
ax1.set_title(verified_str)  # 제목 설정
ax1.axis("off")  # 축 제거

# 두 번째 이미지 표시
ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
ax2.set_title(distance_str)  # 제목 설정
ax2.axis("off")  # 축 제거

# 플롯 표시
plt.tight_layout()  # 레이아웃 조정
plt.show()


time.sleep(100)