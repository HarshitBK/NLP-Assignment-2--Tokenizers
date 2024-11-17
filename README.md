# **Assignment: Tokenizer and Model Training**

This assignment involves training tokenizers, evaluating their performance, and training a model under specific constraints. The tasks and methodologies are outlined below.

---
# Contribution Table

| **Contributor(s)**                    | **Task**                                                                                           
|---------------------------------------|----------------------------------------------------------------------------------------------------
| **Harshit**                           | - Show the matrix with the fertility score and dataset size.                                        
|                                       | - Show the matrix with perplexity for each epoch and test the model's output for 10 prompts.        
| **Hirva, Harshit, Khushal, Harshi**   | - Train 5 Tokenizers on five samples from the dataset you had scraped from the earlier assignment.  
|                                       | - Calculate the fertility score of all the five Tokenizers that you have trained.                   
| **Khushal and Harshi**                | - Tokenize your dataset using the best tokenizer you trained in Task 1.                             
|                                       | - Train your model using the tokenized dataset. Note down the perplexity of your model for every 0.1 epoch. 
| **Rushi**                             | - Choose any of the predefined model architectures & adjust it in such a way that its total parameters are less than 100M. 
|                                       | - Reporting the results and Documentation.                                                             


## **Task 1: Tokenizer Training**

### **Objective**
Train five different tokenizers on samples of the dataset and evaluate them based on fertility scores.

### **Tokenizers**

1. **Whitespace Tokenizer**
   - **Description**: Splits text into tokens based on spaces, tabs, or newlines.
   - **Fertility Scores**: Ranged from **1.00** to **1.319**.
   - **Observation**: The scores are relatively stable, with **Part 1** showing the highest fertility score. The Whitespace Tokenizer produces a moderate number of tokens, similar to SentencePiece.
   - **Advantages**: Simple and easy to implement.
   - **Disadvantages**: 
     - Lacks subword-level tokenization.
     - Fails to capture linguistic nuances, making it unsuitable for complex NLP tasks.

2. **Character-level Tokenizer**
   - **Description**: Breaks text into individual characters, including letters, punctuation, and spaces.
   - **Fertility Scores**: Ranged from **1.00** to **1.314**.
   - **Observation**: The Character-level Tokenizer shows consistent fertility scores with some variation across parts. The results are similar to those of the Whitespace Tokenizer, suggesting that it breaks text into smaller units with moderate efficiency.
   - **Advantages**: 
     - Highly flexible.
     - Useful for tasks requiring fine-grained analysis, such as spelling correction or languages with small alphabets.
   - **Disadvantages**: 
     - Generates very long sequences, leading to inefficiency.
     - Struggles to capture meaningful context.

3. **Byte-Pair Encoding (BPE) Tokenizer**
   - **Description**: Uses a data-driven approach to iteratively merge frequent character pairs into subwords.
   - **Fertility Scores**: Ranged from **3.01** to **3.91** across all five parts.
   - **Observation**: Byte Level Tokenizer generates more tokens, resulting in higher fertility scores. The scores increase across parts, which may lead to longer sequences, potentially affecting efficiency.
   - **Advantages**: 
     - Effectively reduces sequence lengths.
     - Handles unseen words flexibly.
   - **Disadvantages**: 
     - Requires a pre-trained vocabulary.
     - Fixed merging rules may not adapt well to new datasets.

4. **SentencePiece Tokenizer**
   - **Description**: Operates on raw text and treats spaces as tokens, making it language-agnostic.
   - **Fertility Scores**: Ranged from **1.013** to **1.372** across the five parts.
   - **Observation**: The fertility scores are moderate and relatively stable. **Part 1** has the highest score, while subsequent parts show slightly lower scores. This indicates a balanced tokenization process.
   - **Advantages**: 
     - Compact and versatile in subword tokenization.
     - Handles unseen data effectively.
   - **Disadvantages**: 
     - May produce subwords that lack linguistic significance.
     - May require additional preprocessing.

5. **WordPiece Tokenizer**
   - **Description**: Segments text into meaningful subwords based on vocabulary likelihood.
   - **Fertility Scores**: Ranged from **0.0005** to **0.567**.
   - **Observation**: WordPiece produces the lowest fertility scores, particularly in **Part 2** and **Part 3**, with extremely low scores (close to **0.0005**). This indicates that WordPiece is highly efficient at tokenizing, producing very few tokens for the input text.
   - **Advantages**: 
     - Balances vocabulary size and sequence length efficiently.
     - Robust in handling rare or unseen words by splitting them into subwords.
     - Widely used in successful models like BERT.
   - **Disadvantages**: None significant for general NLP tasks.


## Conclusion
- **Most Efficient**: WordPiece tokenizer, with the lowest fertility scores.
- **Least Efficient**: Byte Level Tokenizer, which generates the most tokens.


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

# Model Training Perplexity Log

## Overview of Perplexity Changes

The following observations highlight the changes in **perplexity** over the course of the 3 epochs of training. Perplexity is a measure used to evaluate the performance of a language model, where lower values indicate better model performance. As perplexity decreases, it shows that the model is learning better representations of the data.

### Epoch 1
- The perplexity starts at **109.45** at step 1 and decreases to **96.23** by step 10. This reflects the model's initial phase of learning, where it gradually adapts to the dataset.

### Epoch 2
- Perplexity continues to drop significantly during the second epoch, starting at **95.67** in step 1 and finishing at **81.23** by step 10. This decrease indicates that the model is refining its predictions and getting better at capturing patterns in the data.

### Epoch 3
- By the third epoch, the model's perplexity reaches its lowest values, starting at **79.67** in step 1 and ending at **70.34** by step 10. This demonstrates that the model is nearing convergence and has effectively learned from the dataset.

## Observations

- **Decreasing Perplexity**: As the number of epochs increases, the model's perplexity consistently decreases, which indicates an improvement in the language model’s ability to predict words. This is expected behavior during training as the model learns more from the data over time.

- **Training Progress**: From the first to the third epoch, the perplexity reduces significantly, indicating that the model is becoming more accurate with each epoch. The reduction in perplexity is a sign of improved generalization and better language understanding.

## Conclusion

Increasing the number of epochs results in progressively lower perplexity values. This suggests that with more training, the model is becoming more accurate in its predictions, capturing linguistic structures, and adapting to the dataset. A lower perplexity value at the end of the third epoch reflects the model’s ability to better understand the text and predict the next words with higher confidence.


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

## **Note**:- The tokenized dataset is uploaded to the server

