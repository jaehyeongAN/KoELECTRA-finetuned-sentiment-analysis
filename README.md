# KoELECTRA-finetuned-sentiment-analysis 😊😐😥🤬
본 모델은 특정 corpus에 편향된 감성분석이 아닌 **일반화되고 범용성 높은 감성분석**을 수행하고자 fine-tuning된 모델입니다.  
(새로운 corpus 등장 시 추가 학습 예정!😎)
<br/>

Training을 위해 [monologg](https://github.com/monologg)님의 [KoELECTRA](https://github.com/monologg/KoELECTRA) 모델을 [bab2min](https://github.com/bab2min)님께서 공유해주신 [naver-shopping-review corpus](https://github.com/bab2min/corpus/blob/master/sentiment/naver_shopping.txt) 및 [steam-game-review corpus](https://github.com/bab2min/corpus/blob/master/sentiment/steam.txt)와 [Lucy Park](https://github.com/e9t)님께서 공유해주신 [NSMT](https://github.com/e9t/nsmc) 데이터 셋을 활용하여 fine-tuning을 진행하였습니다.  
</br>

## 📊 Evaluation
각 corpus별 성능 평가 결과는 아래와 같습니다.  
**1. naver shopping reivew**
```
              precision    recall  f1-score   support
           0       0.97      0.93      0.95     19975
           1       0.93      0.97      0.95     20025
    accuracy                           0.95     40000
```
>naver-shopping-review 코퍼스에 대해 해당 단일 코퍼스로 모델 학습 및 성능 평가 시 96% acc를 보인 것에 비해  
>서로 다른 분야의 sentiment corpus를 학습하였음에도 95% acc로 성능이 크게 하락하지 않았습니다!👏👏

**2. naver sentiment movie corpus(NSMT)**
```
              precision    recall  f1-score   support
           0       0.96      0.89      0.93     20116
           1       0.90      0.96      0.93     19883
    accuracy                           0.93     39999
```
**3. steam game review**
```
              precision    recall  f1-score   support
           0       0.88      0.88      0.88      9973
           1       0.88      0.88      0.88     10027
    accuracy                           0.88     20000
```  
</br>

## ✏ Usage
본 모델은 🤗**huggingface transformers**🤗에 porting 되어 있으며 아래와 같이 쉽게 weight을 직접 다운로드 없이 사용하실 수 있습니다.

#### 1. Install pytorch and transformers
```bash
$ pip install torch
$ pip install transformers
```  


#### 2. Load transformers
```python
# import library
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# load model
tokenizer = AutoTokenizer.from_pretrained("jaehyeong/koelectra-base-v3-finetuned-generalized-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("jaehyeong/koelectra-base-v3-finetuned-generalized-sentiment-analysis")
sentiment_classifier = pipeline('sentiment-analysis', tokenizer=tokenizer, model=model)
```  


#### 3.Predict
```python
review_list = [
	'이쁘고 좋아요~~~씻기도 편하고 아이고 이쁘다고 자기방에 갖다놓고 잘써요~^^',
	'아직 입어보진 않았지만 굉장히 가벼워요~~ 다른 리뷰처럼 어깡이 좀 되네요ㅋ 만족합니다. 엄청 빠른발송 감사드려요 :)',
	'재구매 한건데 너무너무 가성비인거 같아요!! 다음에 또 생각나면 3개째 또 살듯..ㅎㅎ',
	'가습량이 너무 적어요. 방이 작지 않다면 무조건 큰걸로구매하세요. 물량도 조금밖에 안들어가서 쓰기도 불편함',
	'한번입었는데 옆에 봉제선 다 풀리고 실밥도 계속 나옵니다. 마감 처리 너무 엉망 아닌가요?',
	'따뜻하고 좋긴한데 배송이 느려요',
	'맛은 있는데 가격이 있는 편이에요'
]

for idx, review in enumerate(review_list):
  pred = sentiment_classifier(review)
  print(f'{review}\n>> {pred[0]}')
```
```
이쁘고 좋아요~~~씻기도 편하고 아이고 이쁘다고 자기방에 갖다놓고 잘써요~^^
>> {'label': '1', 'score': 0.9945501685142517}
아직 입어보진 않았지만 굉장히 가벼워요~~ 다른 리뷰처럼 어깡이 좀 되네요ㅋ 만족합니다. 엄청 빠른발송 감사드려요 :)
>> {'label': '1', 'score': 0.995430588722229}
재구매 한건데 너무너무 가성비인거 같아요!! 다음에 또 생각나면 3개째 또 살듯..ㅎㅎ
>> {'label': '1', 'score': 0.9959582686424255}
가습량이 너무 적어요. 방이 작지 않다면 무조건 큰걸로구매하세요. 물량도 조금밖에 안들어가서 쓰기도 불편함
>> {'label': '0', 'score': 0.9984619617462158}
한번입었는데 옆에 봉제선 다 풀리고 실밥도 계속 나옵니다. 마감 처리 너무 엉망 아닌가요?
>> {'label': '0', 'score': 0.9991756677627563}
따뜻하고 좋긴한데 배송이 느려요
>> {'label': '1', 'score': 0.6473883390426636}
맛은 있는데 가격이 있는 편이에요
>> {'label': '1', 'score': 0.5128092169761658}
```
- label 0 : negative review
- label 1 : positive review
