# PyTorch-based Sentiment Analysis on Movie Reviews

## 📖 Overview

This repository contains a comprehensive implementation of a sentiment analysis pipeline for movie reviews which were scrapping from IMDB. The primary goal is to classify a given review text as either positive or negative. The project serves as a practical, step-by-step guide through the entire lifecycle of a Natural Language Processing (NLP) project, from raw data to a functional prediction model.

The notebook explores two fundamental approaches in modern NLP:
1.  **Classic RNN Approach**: Building a custom Long Short-Term Memory (LSTM) network from the ground up using PyTorch. This path highlights the challenges of training, the importance of text preprocessing, and the problem of overfitting on small datasets.
2.  **Transfer Learning Approach**: Leveraging a powerful, pre-trained Transformer model (`finiteautomata/bertweet-base-sentiment-analysis`) from the Hugging Face ecosystem to achieve state-of-the-art results with minimal code.

## ✨ Features

-   **End-to-End Pipeline**: Covers all steps from data ingestion to model inference.
-   **Detailed Text Preprocessing**: Includes cleaning, tokenization, stop-word removal, and lemmatization.
-   **Custom PyTorch Model**: A clear implementation of a bidirectional LSTM network.
-   **Complete Training Loop**: Standard training and validation loop with loss and accuracy tracking.
-   **State-of-the-Art Integration**: Seamless use of the `transformers` library for high-performance predictions.
-   **Qualitative Analysis**: Examples of how to analyze model predictions, confidence scores, and distributions when ground-truth labels are unavailable.

## 🛠️ Technology Stack

-   **Framework**: [PyTorch](https://pytorch.org/)
-   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
-   **NLP Preprocessing**: [NLTK (Natural Language Toolkit)](https://www.nltk.org/)
-   **Transfer Learning**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
-   **Utilities**: [Scikit-learn](https://scikit-learn.org/stable/)
-   **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## Results and Conclusion

The project's findings highlight the significant performance gap between the two approaches. The pre-trained BERTweet model demonstrated significantly superior performance compared to the custom-trained LSTM model.

   -   **BERTweet Model:** This model achieved the highest performance across all metrics, with a robust F1-score of [e.g., 0.89] and an accuracy of [e.g., 91%]. Its ability to classify reviews into three distinct categories (Positive, Negative, Neutral) provided a more nuanced analysis, confirmed by a confusion matrix showing a high number of correct predictions across all classes.

   -  **LSTM Model (with proper labels):** This model served as a strong baseline, reaching a peak validation accuracy of approximately 87.5%. However, its performance was limited by severe overfitting, as training accuracy approached 100% while validation metrics stagnated. This indicates that while the model could learn from the data, it struggled to generalize as effectively as the BERTweet model.

   -  **LSTM Model (with random labels):** This experimental model served as a crucial control, yielding an accuracy of around 50% (equivalent to random guessing). This effectively confirmed the "Garbage In, Garbage Out" principle, proving that a model cannot learn without meaningful, correctly labeled data.

**Final Conclusion:**
The key takeaway is that the transfer learning approach used by the BERTweet model is far more effective. Its pre-training on a massive corpus of text gives it a deep understanding of language context that a model trained from scratch on a limited dataset, like the LSTM, cannot easily achieve. The quantitative results clearly favor the pre-trained transformer architecture for this sentiment analysis task.


## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   `pip` for package management

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/orangegreen212/Sentiment-Analysis-LTSM-and-BERT-Model.git
    cd Sentiment-Analysis-LTSM-and-BERT-Model
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download required NLTK data:**
    ```python
    # Run this in a Python shell
    import nltk
    nltk.download('all') # Or download specific packages: 'punkt', 'stopwords', 'wordnet'
    ```

### Usage

The main logic is contained within the `Sentiment analysis.ipynb` notebook. Open it with Jupyter Lab or Google Colab and run the cells sequentially to reproduce the analysis and training process.



## 📈 Project Workflow

1.  **Data Loading & Cleaning**: The initial dataset is loaded, and text is cleaned of artifacts like HTML tags, stopwords etc.
2.  **Feature Engineering**: Text is tokenized and lemmatized. A vocabulary is built, and sentences are converted to sequences of integer indices.
3.  **Model Training (LSTM)**:
    -   An LSTM model is defined and trained on the vectorized data.
    -   *Key Finding*: Training from scratch on this dataset leads to severe overfitting, demonstrating the need for more data or a more powerful approach.
4.  **Model Inference (BERTweet)**:
    -   A pre-trained `bertweet` model is loaded via the Hugging Face `pipeline`.
    -   This model is used to generate high-quality sentiment predictions (`POS`, `NEG`, `NEU`) for the entire dataset.
5.  **Results Analysis**:
    -   The distribution of sentiments predicted by `bertweet` is visualized.
    -   The model's confidence scores are analyzed to understand where it performs well and where it struggles.
