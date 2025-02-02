# Sentiment Analysis using BERT

This repository implements a transformer model using Bidirectional Encoder Representations from Transformers (BERT) for sentiment analysis based on textual reviews and star ratings. The project is the culmination of a final project for the Machine Learning Course.

## Project Overview

### Objective
The goal of this project is to classify textual reviews into positive, negative, or neutral sentiments based on both the text review and associated star ratings. The approach fine-tunes a pre-trained BERT model to enhance its ability to discern specific linguistic characteristics and sentiment expressions.

### Key Features
- **Transformer Model**: Utilizes the BERT architecture for language understanding and sentiment classification.
- **Custom Layers and Fine-Tuning**: Modifies the final layers of BERT and fine-tunes it for the unique linguistic characteristics of Yelp reviews.
- **Specialized Training Strategy**: Balances the broad language understanding of BERT with the nuances of consumer feedback in the service industry.
- **Custom Data Loaders**: Develops tailored data loaders and preprocessing pipelines to handle the informal and diverse styles of Yelp reviews.
- **Performance Validation**: Rigorous validation using standard evaluation metrics demonstrates the model's superior performance over baseline models.

## Methodology
### 1. Customization for Yelp Reviews
- Adapted BERT specifically for Yelp reviews, addressing their informal linguistic styles and unique characteristics.
- Developed tailored data loaders and preprocessing pipelines to align with the structure of the Yelp dataset.

### 2. Model Architecture Modification
- Strategically modified the last few layers of the BERT model to improve sentiment classification performance.
- Enhancements focus on the unique sentiment expressions encountered in Yelp reviews.

### 3. Focused Training Strategy
- Designed a specialized training loop to balance pre-trained knowledge with task-specific learning.
- Fine-tuned BERT to optimize performance for Yelp reviews, rather than general sentiment analysis.

### 4. Validation
- Evaluated the model using standard metrics, ensuring robustness and reliability in sentiment classification.

## Results
The modified BERT model achieved a significant improvement in sentiment classification accuracy compared to baseline models. This project highlights the critical role of task-specific fine-tuning in natural language processing applications.

## File Structure
- **`Sentiment_analysis_notebook.ipynb`**: Full implementation, including data processing, model fine-tuning, evaluation, and visualization.
- **`data/`**: Placeholder for input data (reviews and star ratings).
- **`models/`**: Directory for saving trained model checkpoints.
- **`results/`**: Directory for storing evaluation results and visualizations.

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install torch transformers pandas numpy matplotlib
```

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Sentiment_analysis_for_YELP_reviews.git
   cd Sentiment_analysis_for_YELP_reviews
   ```

2. **Prepare Data**:
   - Place your dataset (reviews and ratings) in the `data/` directory.

3. **Run the Notebook**:
   - Open `Sentiment_analysis_notebook.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Follow the steps to process data, fine-tune the model, and evaluate performance.

4. **View Results**:
   - Check the `results/` directory for evaluation metrics and visualizations.

## Example Usage

```python
# Example snippet from the notebook:
from transformers import BertForSequenceClassification, BertTokenizer

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize input text
inputs = tokenizer("This movie was great!", return_tensors="pt")
outputs = model(**inputs)
```

