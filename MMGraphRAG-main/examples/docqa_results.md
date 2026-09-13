# 文档问答评估报告

**生成时间**: 2026-01-07 20:12:12  
**问题数量**: 10  
**知识图谱**: `examples/example_output`

---

## 问题 1

**类型**: `text-only`

### 问题
How does the paper propose to calculate the coefficient α for the Weighted Cross Entropy Loss?

### 标准答案
The coefficient α is calculated using the equation lg((n-nt)/(nt+K)), where nt is the number of samples with class t, n is the total number of samples in the training set, and K is a hyperparameter to tune.

### 依据说明
"In this work, we use lg((n-nt)/(nt+K)) to calculate the coefficient α..."

### 模型回答
The paper proposes to calculate the coefficient αᵢ for the Weighted Cross-Entropy Loss using a **class-based inverse frequency weighting scheme** with a logarithmic transformation. Specifically, the formula is:

\[
\alpha_i = \log\left(\frac{n - n_t}{n_t} + K\right)
\]

where:  
- \(n_t\) is the number of samples in class \(t\) (the class of the i-th sample),  
- \(n\) is the total number of training samples,  
- \(K\) is a tunable hyperparameter added for numerical stability and to control the magnitude of the weights.

This formulation assigns **higher weights to samples from minority classes** (where \(n_t\) is small), thereby increasing their contribution to the loss and helping to mitigate class imbalance during training. The coefficient αᵢ is incorporated into the weighted cross-entropy loss as:

\[
\mathrm{WCE} = -\frac{1}{N} \sum_{i} \alpha_i \sum_{j \in \{0,1\}} y_{ij} \log p_{ij}
\]

The use of the logarithm softens extreme weight values while still emphasizing rare classes, and the hyperparameter \(K\) prevents division-by-zero or excessively large weights when \(n_t\) is very small.

---

## 问题 2

**类型**: `text-only`

### 问题
Is the OntoNotes4.0 dataset used for the Named Entity Recognition task Chinese or English?

### 标准答案
Chinese.

### 依据说明
The specific part in the text mentioning "Chinese OntoNotes4.0 is a Chinese dataset" confirms the language of the dataset.

### 模型回答
The **OntoNotes4.0 dataset** itself is **not exclusively Chinese or English**—it is a **multilingual corpus** that includes data in multiple languages, including **both English and Chinese**. However, in the context of the provided data:

- The entry **"CHINESE ONTONOTES4.0"** (and its variants like "CHINESE ONTONOTES4.0 NER DATASET") explicitly refers to the **Chinese-language portion** of the OntoNotes 4.0 corpus.
- This Chinese subset is used specifically for **Named Entity Recognition (NER)** tasks in several experiments cited, such as those involving BERT-MRC, Lattice-LSTM, and Tversky Index hyperparameter tuning.
- The description clarifies: *"Chinese OntoNotes4.0 is a named entity recognition dataset consisting of news-domain texts with 18 entity types"* and that it is the **Chinese version** of the OntoNotes corpus.

Meanwhile:
- There is also an **English OntoNotes5.0** mentioned in the sources (used alongside CoNLL2003), but **OntoNotes4.0 in this context predominantly refers to the Chinese variant** when discussed in relation to NER experiments in the provided tables and text.

### Conclusion:
In the specific context of the provided data and experiments, **OntoNotes4.0 is used as a Chinese Named Entity Recognition dataset**. The broader OntoNotes 4.0 corpus includes multiple languages, but the referenced usage here is **Chinese**.

---

## 问题 3

**类型**: `multimodal-t`

### 问题
Which model variant has the highest improvement in F1 score for the QuoRef dataset when compared to the base XLNet model?

### 标准答案
The model variant with the highest improvement in F1 score for the QuoRef dataset compared to the base XLNet model is XLNet+DSC, with a +1.41 increase.

### 依据说明
The table shows the improvement in parentheses, and the base XLNet model's score on QuoRef is compared to each variant model, with XLNet+DSC showing the highest positive difference in performance in the F1 score.

### 模型回答
The **XLNet+DSC** model variant achieves the highest improvement in F1 score for the **QuoRef dataset**, with a **+1.41** increase over the base XLNet model, reaching an F1 score of **72.90**. This is the only XLNet variant reported to show improvement on QuoRef in the provided data, making it the top-performing variant by default. The improvement is attributed to the application of **Discriminative Self-Consistency (DSC)**, which enhances performance on complex reasoning tasks like those in QuoRef.

---

## 问题 4

**类型**: `multimodal-t`

### 问题
What data augmentation method resulted in the highest F1-score for the BERT model according to Table 8?

### 标准答案
The "+ positive & negative" data augmentation method resulted in the highest F1-score for the BERT model with a score of 93.14.

### 依据说明
The question asks for the maximum value in a single row, which is straightforward to find by looking at the BERT row across all data augmentation methods.

### 模型回答
According to Table 8, the data augmentation method that resulted in the highest F1-score for the BERT model was **"+ positive & negative"**, achieving an F1-score of **93.14**. This approach augmented the training data with both additional positive and negative examples while maintaining a balanced class distribution (50% positive, 50% negative), leading to the best performance among all tested methods.

---

## 问题 5

**类型**: `multimodal-t`

### 问题
What is the highest F1 score achieved on the Chinese OntoNotes4.0 dataset, according to Table 10?

### 标准答案
The highest F1 score on the Chinese OntoNotes4.0 dataset is 84.67.

### 依据说明
The question requires identifying the maximum value in a single column, which is straightforward.

### 模型回答
According to **Table 10**, the **highest F1 score** achieved on the **Chinese OntoNotes4.0 NER dataset** is **84.67**, which occurs when the Tversky Index hyperparameter **α = 0.6**.

This result is part of an experiment evaluating the effect of the α hyperparameter in the Tversky Index (with β set to 1 − α). The scores across different α values show that performance peaks at α = 0.6 for this dataset, highlighting the sensitivity of model performance to the choice of α in imbalanced learning scenarios such as named entity recognition.

---

## 问题 6

**类型**: `multimodal-t`

### 问题
Which model achieved the highest F1 score in the English WSJ dataset?

### 标准答案
BERT-Tagger+DSC achieved the highest F1 score in the English WSJ dataset with a score of 99.38.

### 依据说明
The question requires examining only the F1 scores in the English WSJ section and identifying the highest one.

### 模型回答
### Highest F1 Score on the English WSJ Dataset

Based on **Table 4** from the provided data (as referenced in the *Sources* and linked via relationships to *"BOHNET ET AL., 2018"* and various model entries), the **English WSJ dataset** results include the following models and their F1 scores:

| Model                              | F1 Score |
|-----------------------------------|----------|
| Meta BiLSTM (Bohnet et al., 2018) | 98.23    |
| BERT-Tagger (Devlin et al., 2018) | 98.86    |
| BERT-Tagger + FL                  | 98.88 (+0.02) |
| BERT-Tagger + DL                  | 98.91 (+0.05) |
| **BERT-Tagger + DSC**             | **99.38 (+0.52)** |

> **Note**: The values in parentheses (e.g., +0.52) indicate absolute improvement over the baseline BERT-Tagger.

### Conclusion

The **BERT-Tagger + DSC** model achieved the **highest F1 score of 99.38** on the English WSJ dataset, outperforming all other listed models, including the original BERT-Tagger and the Meta BiLSTM by Bohnet et al. (2018).

This result highlights the effectiveness of incorporating the **Dice Similarity Coefficient (DSC)** as a loss function (or optimization target), which—given its equivalence to the F1 score—directly optimizes for this evaluation metric.

---

## 问题 7

**类型**: `multimodal-t`

### 问题
What performance boost did BERT+DSC achieve for the MRPC?

### 标准答案
BERT+DSC achieved a performance boost of +0.92 for the MRPC.

### 依据说明
In Table 7, it is shown next to BERT+DSC in the MRPC column.

### 模型回答
BERT+DSC achieved a **+0.92 absolute improvement in F1 score** on the MRPC (Microsoft Research Paraphrase Corpus) task compared to the base BERT model. Specifically:

- **Base BERT F1 score**: 88.0  
- **BERT+DSC F1 score**: 88.92  
- **Performance boost**: **+0.92 F1 points**

This improvement is the largest among the evaluated enhancements (FL, DL, and DSC) for BERT on this benchmark.

---

## 问题 8

**类型**: `multimodal-t`

### 问题
How does the performance of BERT+DL on SST-5 compare to that of BERT+DSC?

### 标准答案
BERT+DL performs worse on SST-5 with an accuracy of 54.63 compared to BERT+DSC which has an accuracy of 55.19.

### 依据说明
Direct comparison between the BERT+DL and BERT+DSC accuracy numbers in the SST-5 column shows that BERT+DL has a lower accuracy.

### 模型回答
On the SST-5 dataset, **BERT+DSC outperforms BERT+DL** in terms of accuracy:

- **BERT+DL** achieves an accuracy of **54.63%**  
- **BERT+DSC** achieves an accuracy of **55.19%**

This represents a **0.56 percentage point improvement** for BERT+DSC over BERT+DL. However, both models are slightly outperformed by **BERT+CE**, which reaches **55.57%** accuracy—the highest among the three. The relatively lower performance of dice-based losses (DL and DSC) compared to cross-entropy (CE) is consistent with their design objective: optimizing F1-score-like metrics rather than accuracy.

---

## 问题 9

**类型**: `multimodal-t`

### 问题
By how much does the accuracy of BERT+CE on SST-2 exceed that of BERT+DL?

### 标准答案
The accuracy of BERT+CE on SST-2 exceeds that of BERT+DL by 0.53.

### 依据说明
Subtraction of the BERT+DL accuracy (94.37) from the BERT+CE accuracy (94.90) on SST-2 gives the difference of 0.53.

### 模型回答
The accuracy of BERT+CE on SST-2 exceeds that of BERT+DL by **0.53 percentage points**.

- BERT+CE accuracy: **94.90%**  
- BERT+DL accuracy: **94.37%**  
- Difference: **94.90 − 94.37 = 0.53**

---

## 问题 10

**类型**: `multimodal-t`

### 问题
What is the ratio of negative to positive examples for the Quoref task?

### 标准答案
The ratio of negative to positive examples for the Quoref task is 169.

### 依据说明
This information is directly stated in the table under the column "ratio" for the Quoref task.

### 模型回答
The ratio of negative to positive examples for the QUOREF task is **169**. This means that for every positive example in the dataset, there are approximately 169 negative examples.

This figure is explicitly reported in **Table 1** of the source material, which provides data imbalance statistics across several NLP tasks. Specifically, for QUOREF (Dasigi et al., 2019), the dataset contains:
- **6.52 million** negative examples,
- **38.6 thousand** positive examples,

which yields a **negative-to-positive ratio of 169**. This extreme imbalance is typical in machine reading comprehension tasks, where only a small fraction of tokens correspond to answer spans (positive examples), and the vast majority are non-answer (negative) tokens.

---

