# Financial Sentiment Analysis

This project fine-tunes a DistilBERT model on the Financial PhraseBank dataset to classify financial news sentences as positive, negative, or neutral. The model achieves 86.75% accuracy on the test set, significantly outperforming the baseline model.

## Project Overview

Financial sentiment analysis has important applications in algorithmic trading, market monitoring, and financial news analysis. This project demonstrates how to:

1. Prepare the Financial PhraseBank dataset
2. Fine-tune a pre-trained language model (DistilBERT)
3. Optimize hyperparameters for the best performance
4. Evaluate the model and analyze errors
5. Create an inference pipeline for practical use

## Setup Instructions

### Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/KaviprakashRamalingam/Fine-tuning-LLM
   cd Fine-tuning-LLM
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Google Colab Setup

If you prefer using Google Colab:

1. Open a new Colab notebook
2. Install the required packages:
   ```python
   !pip install transformers datasets accelerate evaluate scikit-learn pandas matplotlib seaborn gradio
   ```
3. Upload the project files or clone the repository:
   ```python
   !git clone https://github.com/KaviprakashRamalingam/Fine-tuning-LLM.git
   %cd Fine-tuning-LLM
   ```

## Dataset

The Financial PhraseBank dataset (Malo et al., 2014) contains sentences from financial news articles labeled with sentiment. The project uses the "sentences_allagree" variant where all annotators agreed on the sentiment label.

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
```

## Training the Model

Execute the Jupyter notebook:

```bash
jupyter notebook financial_sentiment_training.ipynb
```

or Open Google Colab

Find the appropriate section to Train your model and run it

## Using the Model

### Command Line Interface

### Python API

```python
from inference import analyze_financial_sentiment

result = analyze_financial_sentiment("The company reported strong quarterly earnings.")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Web Interface

Start the Gradio web interface in Google colab:
Open the URL mentioned in the output of the cell

```
Example: 
* Running on public URL: https://0f4506f07fd5a9661a.gradio.live
```

In case of Streamlit, 

Run the streamlit cell in your google colab and it gives you three localhost IPs

```
⠙⠹⠸⠼⠴⠦⠧⠇⠏
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.28.0.12:8501
  External URL: http://34.106.81.117:8501
```

The app runs at a different URL

```
⠋⠙⠹your url is: https://fair-things-wave.loca.lt
```

On opening this URL, you will be asked to provide a Tunnel password in which paste the IP obtained in the External URL above excluding the port number

```
34.106.81.117
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Baseline | 0.6423 | 0.5924 | 0.6423 | 0.5841 |
| Fine-tuned | 0.8675 | 0.8702 | 0.8675 | 0.8683 |

## References

1. Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
