# **Assignment: Tokenizer and Model Training**

This assignment involves training tokenizers, evaluating their performance, and training a model under specific constraints. The tasks and methodologies are outlined below.

---

## **Task 1: Tokenizer Training**

### **Objective**
Train five different tokenizers on samples of the dataset and evaluate them based on fertility scores.

### **Tokenizers**

1. **Whitespace Tokenizer**
   - **Description**: Splits text into tokens based on spaces, tabs, or newlines.
   - **Advantages**: Simple and easy to implement.
   - **Disadvantages**: 
     - Lacks subword-level tokenization.
     - Fails to capture linguistic nuances, making it unsuitable for complex NLP tasks.

2. **Character-level Tokenizer**
   - **Description**: Breaks text into individual characters, including letters, punctuation, and spaces.
   - **Advantages**: 
     - Highly flexible.
     - Useful for tasks requiring fine-grained analysis, such as spelling correction or languages with small alphabets.
   - **Disadvantages**: 
     - Generates very long sequences, leading to inefficiency.
     - Struggles to capture meaningful context.

3. **Byte-Pair Encoding (BPE) Tokenizer**
   - **Description**: Uses a data-driven approach to iteratively merge frequent character pairs into subwords.
   - **Advantages**: 
     - Effectively reduces sequence lengths.
     - Handles unseen words flexibly.
   - **Disadvantages**: 
     - Requires a pre-trained vocabulary.
     - Fixed merging rules may not adapt well to new datasets.

4. **SentencePiece Tokenizer**
   - **Description**: Operates on raw text and treats spaces as tokens, making it language-agnostic.
   - **Advantages**: 
     - Compact and versatile in subword tokenization.
     - Handles unseen data effectively.
   - **Disadvantages**: 
     - May produce subwords that lack linguistic significance.
     - May require additional preprocessing.

5. **WordPiece Tokenizer**
   - **Description**: Segments text into meaningful subwords based on vocabulary likelihood.
   - **Advantages**: 
     - Balances vocabulary size and sequence length efficiently.
     - Robust in handling rare or unseen words by splitting them into subwords.
     - Widely used in successful models like BERT.
   - **Disadvantages**: None significant for general NLP tasks.

### **Chosen Tokenizer: WordPiece**
We found that the **WordPiece Tokenizer** works best for our dataset due to its ability to efficiently balance vocabulary size and sequence length. By segmenting words into meaningful subwords, WordPiece enables the model to capture more granular semantic representations, especially for rare or unseen words. This flexibility in handling out-of-vocabulary (OOV) words without losing context makes it particularly well-suited for our dataset, which contains varied and complex text patterns. Furthermore, WordPiece's integration into successful models like BERT demonstrates its robustness and effectiveness, making it the optimal choice for our task.

---

## **Task 2: Model Training**

### **Objective**
Train a model with total parameters under 100M using the best tokenizer from Task 1 and evaluate its performance.

### **Steps**

# Model Description

## Overview
This project implements a **BERT-based Masked Language Model (MLM)** with optimizations for efficient training and performance. The model is designed to predict masked words in a text sequence, enabling it to capture contextual word semantics effectively.

---

## Key Features

1. **BERT Architecture**:
   - Custom configuration with:
     - **8 hidden layers**
     - **200 hidden size**
     - **8 attention heads**
     - Maximum position embeddings: **512**
   - Vocabulary size adapted to the tokenizer used.
   - Total number of parameters = 66 Million

2. **LoRA (Low-Rank Adaptation)**:
   - Applies low-rank updates to attention layers (`query` and `value`).
   - Drastically reduces the number of trainable parameters while maintaining model performance.

3. **Mixed-Precision Training**:
   - Uses **float16** for computations where possible, reducing memory usage and speeding up training.
   - Ensures stability using **torch.cuda.amp.GradScaler**.

4. **Optimizer and Scheduler**:
   - **AdamW** optimizer with a linear scheduler.
   - Gradual learning rate adjustment for better convergence.

---

## Training Process
- **Dataset**: Tokenized text sequences with attention masks.
- **Training Steps**:
  1. Forward pass to predict masked tokens and compute loss.
  2. Backward pass to update trainable parameters (LoRA layers).
  3. Periodic perplexity calculation to monitor performance.

- **Epochs**: The model is trained for **3 epochs** with perplexity logged every **0.1 epoch**.

---

## Outputs
- **Final Model**: Saved in the `quantized_lora_bert` directory.
- **Perplexity Log**: Captured in `perplexity_log.txt` for detailed performance tracking.

---

## **Model Evaluation: Prompts and Predictions**

| **Prompt**                                                      | **Predicted Next Word** | **Perplexity** |
|-----------------------------------------------------------------|-------------------------|----------------|
| "ਪ੍ਰाकृतिक ਆਪਦਾਵਾਂ ਦੇ ਬਾਵਜੂਦ, ਲੋਕਾਂ ਨੇ..."                      | "ਮਦਦ"                  | 105.4          |
| "ਇਕ ਕਵੀ ਦੀ ਸਹੀ ਮਹੱਤਤਾ ਸਮਾਜ ਵਿੱਚ..."                           | "ਹੁੰਦੀ"                | 98.3           |
| "ਇਹ ਨਵਾਂ ਕਾਨੂੰਨ ਕਿਸੇ ਦੇ ਹੱਕਾਂ ਨੂੰ..."                           | "ਹਾਣੀ"                 | 110.2          |
| "ਜਦੋਂ ਤੁਸੀਂ ਸੱਚਾਈ ਨੂੰ ਦਬਾਉਂਦੇ ਹੋ, ਤਾਂ..."                     | "ਮੁਸ਼ਕਲ"                | 103.6          |
| "ਹੋਰਾਂ ਦੀ ਮਦਦ ਕਰਨਾ ਸਾਡਾ..."                                    | "ਕਰਤੱਬ"               | 92.1           |
| "ਜਦੋਂ ਮੌਸਮ ਬਦਲਦਾ ਹੈ, ਤਦ..."                                    | "ਹਵਾਈ"                | 108.7          |
| "ਇੱਕ ਚੰਗਾ ਵਿਦਿਆਰਥੀ ਹਰ ਹਾਲਤ ਵਿੱਚ..."                           | "ਸਮਝਦਾ"               | 97.9           |
| "ਸਰਕਾਰ ਨੂੰ ਸਿੱਖਿਆ ਦੀ ਖੇਤਰ ਵਿੱਚ..."                             | "ਉਪਲਬਧੀਆਂ"           | 94.2           |
| "ਕਿਸਾਨਾਂ ਦੀ ਸਹਾਇਤਾ ਲਈ ਕਈ ਯੋਜਨਾਵਾਂ..."                         | "ਚਲਾਈ"               | 102.3          |
| "ਜਦੋਂ ਵਿਦਿਆਰਥੀ ਆਪਣੇ ਵਿਸ਼ੇ ਨੂੰ ਪਿਆਰ ਕਰਦੇ ਹਨ, ਉਹ..."            | "ਉਪਲਬਧੀਆਂ"           | 95.4           |


---

## **Deliverables**
1. Code for training and evaluating tokenizers.
2. Fertility score matrix for the five tokenizers.
3. Adjusted model with <100M parameters.
4. Training logs with perplexity scores for every 0.1 epoch.
5. Perplexity matrix and outputs for 10 test prompts.
