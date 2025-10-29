# Animal QTL Paper Classification

This project builds an ensemble NLP model combining **TF-IDF + Linear SVM** and **BioBERT** to classify research papers as *relevant* or *non-relevant* for the Animal QTL database.

## Data
- `QTL_text.json`: Training data (Title, Abstract, Category, PMID)
- `QTL_test_unlabeled.tsv`: Test data (Title, Abstract, PMID)

## Preprocessing
- Remove URLs, emails, and punctuation  
- Remove stopwords (case preserved for BioBERT)  
- Balance dataset by downsampling majority class

## Models
1. **TF-IDF + LinearSVC**  
   - Uses grid search for optimal hyperparameters  
   - Calibrated with sigmoid to produce probabilities  
2. **BioBERT (dmis-lab/biobert-base-cased-v1.1)**  
   - Tokenized (max length 256)  
   - Fine-tuned using Hugging Face Trainer with early stopping  

## Ensemble
Validation F1-scores determine model weights:
