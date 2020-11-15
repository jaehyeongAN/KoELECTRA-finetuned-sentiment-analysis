# KoELECTRA-finetuned-sentiment-analysis

쇼핑 리뷰 감성분석을 위해 [bab2min](https://github.com/bab2min)님께서 공유해주신 [naver-shopping-review corpus](https://github.com/bab2min/corpus/tree/master/sentiment)를 활용하여 [monologg](https://github.com/monologg)님의 [KoELECTRA](https://github.com/monologg/KoELECTRA) pretrained 모델을 fine-tuning한 모델입니다.
<br/>

## Usage
본 모델은 🤗**huggingface transformers**에 porting 되어 있으며 아래와 같이 쉽게 weight을 사용하실 수 있습니다.

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
tokenizer = AutoTokenizer.from_pretrained("jaehyeong/koeletra-base-v3-finetuned-naver-shopping-review-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("jaehyeong/koeletra-base-v3-finetuned-naver-shopping-review-sentiment-analysis")
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
