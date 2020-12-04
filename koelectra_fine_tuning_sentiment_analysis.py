import os

from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AdamW
from transformers import Trainer, TrainingArguments

from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch
from torchsummary import summary

device = torch.device("cuda") # gpu


class SentimentReviewDataset(Dataset):
  
    def __init__(self, dataset):
      self.dataset = dataset
      self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
  
    def __len__(self):
      return len(self.dataset)
  
    def __getitem__(self, idx):
      row = self.dataset.iloc[idx, 0:2].values
      text = row[1]
      y = row[0]

      inputs = self.tokenizer(
          text, 
          return_tensors='pt',
          truncation=True,
          max_length=256,
          pad_to_max_length=True,
          add_special_tokens=True
          )
    
      input_ids = inputs['input_ids'][0]
      attention_mask = inputs['attention_mask'][0]

      return input_ids, attention_mask, y


def merge_review_dataset(data_dir_list):
    # naver shopping review 
    shopping_review_dataset = pd.read_csv(data_dir_list[0], sep='\t', names=['star','review'])
    shopping_review_dataset['star'] = [1 if data['star'] >= 4 else 0 for idx, data in shopping_review_dataset.iterrows()]
    shopping_review_dataset.rename(columns = {"star": "label"}, inplace=True)

    # naver movie review 
    movie_review_dataset = pd.read_csv(data_dir_list[1], sep='\t')
    movie_review_dataset = movie_review_dataset[['label','document']]
    movie_review_dataset.rename(columns={'document':'review'}, inplace=True)

    # steam game review 
    steam_review_dataset = pd.read_csv(data_dir_list[2], sep='\t', names=['label','review'])

    # merge 
    review_dataset = pd.concat([shopping_review_dataset, movie_review_dataset, steam_review_dataset], ignore_index=True)
    review_dataset.dropna(inplace=True)

    return review_dataset


def main():
    data_dir_list = ['./_data/naver_shopping_review.txt', './_data/naver_movie_ratings.txt', './_data/steam_game_review.txt']
    review_dataset = merge_review_dataset(data_dir_list)

    # split train test
    train_data, test_data = train_test_split(review_dataset, test_size=0.2) 

    # Data Loader
    train_dataset = SentimentReviewDataset(train_data)
    test_dataset = SentimentReviewDataset(test_data)

    # Load Pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)
    
    # Train
    epochs = 3
    batch_size = 16
    optimizer = AdamW(model.parameters(), lr=1e-5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    accuracies = []
    for i in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0

        model.train()
        for input_ids_batch, attention_masks_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_batch).sum()
            total += len(y_batch)

            batches += 1
            if batches % 100 == 0:
                print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
        
        losses.append(total_loss)
        accuracies.append(correct.float() / total)
        print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)

    print(losses, accuracies)


    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    for input_ids_batch, attention_masks_batch, y_batch in test_loader:
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        _, predicted = torch.max(y_pred, 1)
        test_correct += (predicted == y_batch).sum()
        test_total += len(y_batch)

        print("Accuracy:", test_correct.float() / test_total)


    # Save model
    save_model_path = './_weights'
    if os.path.exists(save_model_path) == False:
        os.mkdir(save_model_path)  	
    torch.save(model.state_dict(), os.path.join(save_model_path, "koelectra-base-finetuned-sentiment-analysis.bin"))



if __name__ == "__main__":
    main()