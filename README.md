# Tacademy ASAC 2기 딥러닝 프로젝트
## 나만의 컬러링북 만들기 

<hr>

### INDEX
1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 프로세스](#프로젝트-프로세스)
3. [데이터와 모델](#데이터와-모델)
4. [프로젝트 결과물](#프로젝트-결과물)
5. [화풍 변환 모델 실행](#화풍-변환-모델-실행)
6. 스케치 모델 실행
   - [LDC](#LDC)
   - [Sketch Keras](#Sketch-Keras)
7. [채색 모델 실행](#채색-모델-실행)
8. [참고 문헌](#참고-문헌)

<hr>


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


	![012](https://github.com/cku7808/My_Own_Coloring_Book/assets/66200628/3238225f-b2a3-4cff-88ca-22628538f6fb)


   - 풍경 이미지 : [Kaggle Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
   - 인물 이미지 : [Flickr-Faces-HQ Dataset(FFHQ)](https://github.com/NVlabs/ffhq-dataset)


- 사용 모델

	![DL-3조_고운-_-복사본-002](https://github.com/cku7808/My_Own_Coloring_Book/assets/66200628/68fb00d8-4131-4f45-bb08-3129751ce31c)

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
- Requirements
	```
	cd LDC
	pip install -r requirements.txt
	```
 
 	1. 데이터 폴더 생성
  	```
   cd data
   mkdir TRAINDATA
   cd TRAINDATA
   mkdir edge
   mkdir org
   	```
   
   	2. 이미지 resizing, label 이미지 생성 및 lst 파일 생성
	```
 	cd ../..
 	python image_resize.py
 	python edge_detect.py
 	python make_lst.py
   	```
 	3. dataset.py line 10 - 302
	```
	DATASET_NAMES = [
        ...
	'TRAINDATA',
	'VALDATA',
	'TESTDATA'
	]
	```

	```python
	def dataset_info(dataset_name, is_linux=True):
		if is_linux:
			config={
				...
				}
		else:
			config={
				...
				'TRAINDATA': {'img_height': 512,
                    		'img_width': 512,
                    		'train_list': "train_pair.lst",
                    		'data_dir': 'data/face_train',  # mean_rgb
                    		'yita': 0.2},
				...
				}
	```
- Train
	1. dataset.py line 259  
	```python
	if self.arg.train_data.lower() in ['TRAINDATA']
 	```

	2. main.py line 229 -
 		- choose_test_data : train모드에서는 validation data로 지정
		```python
  		parser.add_argument('--choose_test_data',
                        type=int,
                        default=-1, 
                        help='Choose a dataset for testing: 0 - 8')
  		```
  		- is_testing 설정 및 train data 지정
		```python
  		is_testing = False
  		...
  		TRAIN_DATA = DATASET_NAMES[25] 
  		```
  	3. Start Trianing
  	```
	python main.py
   	```
   
  - Test
	1. main.py line 229 -
 		- choose_test_data : test data 지정
		```python
  		parser.add_argument('--choose_test_data',
                        type=int,
                        default=-1, 
                        help='Choose a dataset for testing: 0 - 8')
  		```
  		- is_testing 설정
		```python
  		is_testing = True
  		```
  	2. Start Trianing
  	```
	python main.py --choose_test_data=-1
   	```

#### Sketch Keras
- Requirements
	```
	cd sketchKeras
	pip install -r requirements.txt
	```
- 이미지 준비
  1. 다운로드 링크(dataset, model)
     https://drive.google.com/drive/folders/1gAkEXwKsVwEFxa-1b0NSadKCjd8Eh9qa?usp=sharing

  2. 데이터 셋 준비(train.py line 22-41)
  ```
  def get_train(x_train_folder_path,y_train_folder_path):
    x_train_file_names = os.listdir(x_train_folder_path)
    y_train_file_names = os.listdir(y_train_folder_path)
    x_train = deque([])
    y_train = deque([])
    new_size = (512, 512)
    print(len(y_train_file_names), len(x_train_file_names))
    for i in x_train_file_names:
        from_mat = cv2.imread(x_train_folder_path + i, cv2.IMREAD_GRAYSCALE)
        from_mat = cv2.resize(from_mat, new_size)
        x_train.append(from_mat)

    for i in y_train_file_names:
        from_mat = cv2.imread(y_train_folder_path + i, cv2.IMREAD_GRAYSCALE)
        from_mat = cv2.resize(from_mat, new_size)
        from_mat = canny(from_mat)
        y_train.append(from_mat)

    return x_train,y_train
  ```
- train
  1. train.py 실행
  ```
  python train.py
  ```
  2. train.py line 82-85
  ```
  with tf.device('/gpu:0'):
    model = load_model('mod.h5')
    print(device_lib.list_local_devices())
    get()
  ```
- test
  - 저장된 모델이용하여 test(google colab)
    1. model load
  ```
  model_masterpiece = load_model('/content/drive/MyDrive/딥러닝 팀플/skech_keras/mod_new.h5')
  ```
    2. 이미지 결과 보기
  ```
  new_size = (512, 512)
img = cv2.imread('/content/aaaaaaa.png' , cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, new_size)


img = np.expand_dims(img, axis=0)


img = img.reshape(-1, 512,512,1)
prediction = model_masterpiece.predict(img)

prediction = prediction.reshape(512,512,1)
cv2_imshow(prediction)
  ```
  
### 채색 모델 실행

### 참고 문헌
