# Sentiment Analysis using BERT

This repository contains a project that implements a transformer model using Bidirectional Encoder Representations from Transformers (BERT) for sentiment analysis based on textual reviews and star ratings. The project is the culmination of a final project for the Machine Learning Course.

## Project Overview

### Objective
The goal of this project is to classify textual reviews into positive, negative, or neutral sentiments based on both the text review and associated star ratings. The approach fine-tunes a pre-trained BERT model to enhance its ability to discern specific linguistic characteristics and sentiment expressions.

### Key Features
- **Transformer Model**: Utilizes the BERT architecture for language understanding and sentiment classification.
- **Custom Layers**: Modifies BERT's final layers to optimize for sentiment analysis tasks.
- **Hyperparameter Tuning**: Experiments with various hyperparameters and architectural changes to maximize performance.
- **Specialized Data Loaders**: Processes and aligns textual review data with the BERT architecture seamlessly.
- **Performance Validation**: Rigorous validation using standard evaluation metrics demonstrates the model's superior performance over baseline models.

## Methodology
1. **Fine-tuning BERT**:
   - Adjusted BERT's final layers to enhance its ability to classify sentiments.
   - Conducted extensive experimentation with hyperparameters to achieve optimal results.
2. **Custom Data Loaders**:
   - Designed specialized data loaders to preprocess and align input data with BERT.
3. **Training**:
   - Implemented multiple training loops to balance BERT's broad language understanding with specific task requirements.
4. **Validation**:
   - Evaluated the model using standard metrics to ensure reliability and robustness.

## Results
The modified BERT model achieved a marked improvement in sentiment classification accuracy compared to baseline models. The project highlights the importance of fine-tuning and provides a valuable framework for applying advanced natural language processing techniques in sentiment analysis.

## File Structure
- **`Sentiment_analysis_notebook.ipynb`**: Contains the full implementation, including data processing, model fine-tuning, evaluation, and visualization.
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
   git clone https://github.com/YourUsername/Sentiment_analysis.git
   cd Sentiment_analysis
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

