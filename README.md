# Tacademy ASAC 2기 딥러닝 프로젝트
## 나만의 컬러링북 만들기 

<hr>

### INDEX
1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 프로세스](#프로젝트-프로세스)
3. [데이터와 모델](#데이터-&-모델)
4. [프로젝트 결과물](#프로젝트-결과물)
5. [화풍 변환 모델 실행](#화풍-변환-모델-실행)
6. 스케치 모델 실행
   - [LDC](#LDC)
   - [Sketch Keras](#Sketch-Keras)
7. [채색 모델 실행](#채색-모델-실행)
8. [참고 문헌](#참고-문헌)


### 프로젝트 개요
컬러링북에 대한 수요는 2010년 이후로 꾸준히 이어지고 있으며, 
아마존에서는 개인이 컬러링 도안을 판매할 수 있도록 플랫폼이 존재하고, 아이디어스에서는 실제로 개인 제작 컬러링북을 판매하고 있는 것을 확인할 수 있었습니다.
이처럼 아마추어 제작 도안에 대한 상품성의 존재를 확인함에 따라 누구든지 본인이 원하는 사진으로 컬러링북 도안을 만들어 취미 생활을 즐길 수 있도록,
AI를 활용한 컬러링북 제작 도안 프로젝트를 기획하였습니다.

해당 프로젝트의 차별점은 '개인화'에 있습니다. 
사진을 기반으로 컬러링북 도안을 생성하는 것에 그치지 않고, 사용자가 원하는 이미지를 원하는 화풍의 컬러링북 도안으로 제작해주고, 원하는 색 조합으로 색칠할 수 있는 가이드까지 제공해주고자 하였습니다.


### 프로젝트 프로세스

![DL_3조-009](https://github.com/cku7808/My_Own_Coloring_Book/assets/66200628/c6a01d9c-8fa1-401d-8ba1-8a5f5ab2945c)

> 해당 프로젝트에서는 인물화와 풍경화 두 종류의 이미지를 학습에 사용합니다. 

- 화풍 변환 단계
	- 화풍을 변환하고자 하는 이미지와 변환할 화풍의 이미지를 선택해 사용자가 최종적으로 도안으로 제작하고자 하는 원본 이미지를 생성하는 단계

- 스케치 단계
	- 화풍이 변환된 이미지에서 스케치라인을 따는 단계로, 최종적으로 컬러링북의 도안을 제작하는 단계 

- 채색 가이드 단계
	- 컬러링북 도안의 색칠 가이드를 제공하는 단계로, 화풍이 변환된 이미지에 사용자가 선택한 색조합으로 재채색한 결과를 반환하는 단계







### 데이터와 모델
- 사용 데이터

  ![011](https://github.com/cku7808/My_Own_Coloring_Book/assets/66200628/70c489dd-efef-4091-9749-c20c456f788f)

   - 풍경 이미지 : [Kaggle Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
   - 인물 이미지 : [Flickr-Faces-HQ Dataset(FFHQ)](https://github.com/NVlabs/ffhq-dataset)


- 사용 모델

	- 화풍 변환 모델 : [Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko)
	- 스케치 모델
		- 인물 : [LDC](https://github.com/xavysp/LDC)
		- 풍경 : [Sketch Keras](https://github.com/lllyasviel/sketchKeras)
	- 채색 가이드 모델 : [PaletteNet](https://github.com/yongzx/PaletteNet-PyTorch)


### 프로젝트 결과물 
- 풍경 

![060](https://github.com/cku7808/My_Own_Coloring_Book/assets/66200628/022b2a22-2156-496f-802d-d89930ee6c30)


### 화풍 변환 모델 실행

### 스케치 모델 실행
#### LDC
#### Sketch Keras

### 채색 모델 실행

### 참고 문헌
