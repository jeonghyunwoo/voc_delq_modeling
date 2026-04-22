# Code Overview

This folder contains the main analysis scripts for the VOC-based delinquency resolution modeling project.

The project was designed to use unstructured collection-call text as an additional signal for predicting whether a delinquent account is likely to be resolved, and to support earlier risk detection and more effective operational prioritization.

---

## 1. Purpose of the Code

The code in this folder covers two main tasks:

- building a domain-adapted tokenizer and text vectorization pipeline for VOC data
- using the vectorized counseling text to develop a delinquency resolution prediction model

Because collection-call notes often contain abbreviations, slang, inconsistent spacing, and counselor-specific writing styles, the workflow was designed to reflect the characteristics of real operational text rather than rely only on generic preprocessing.

---

## 2. Files

### `0_vectorizer_modeling.py`
Builds the text preprocessing and vectorization components used in the project.

Main roles:

- train a domain-adapted tokenizer using **soynlp**, which is useful for VOC text containing slang, abbreviations, and inconsistent spacing
- build a **Doc2Vec**-based vectorizer for counseling text
- save tokenizer / vectorizer artifacts for downstream modeling

This script focuses on turning noisy counseling text into a usable analytical representation.

---

### `1_voc_modeling.py`
Builds the delinquency resolution prediction workflow using vectorized counseling text.

Main roles:

- prepare monthly modeling data from delinquency-related records
- clean and filter counseling text
- generate text features using **TF-IDF** and **Doc2Vec**
- use vectorized counseling text to develop a classification model for **delinquency resolution probability**
- support operational use such as prioritization of collection targets

This script is the core modeling pipeline of the project.

---

## 3. Analytical Flow

The overall code flow can be understood as follows:

1. collect and clean collection-call VOC text
2. train a tokenizer suited to domain-specific expressions
3. convert text into numerical vectors
4. combine text signals with account-level delinquency information
5. train a model to estimate delinquency resolution probability
6. use the result as a practical signal for monitoring and operational prioritization

---

## 4. Public Repository Notes

This folder is part of a simplified public version of an internal project.

The original workflow used internal data sources and environment-specific utilities, so the public version may not include:

- source datasets
- internal DB connection code
- private helper modules
- full execution environment used in production

Accordingly, this folder is intended to show:

- the modeling workflow
- the NLP feature engineering approach
- the role of VOC text in delinquency management
- the structure of the project code

rather than provide a fully reproducible production package.

---

## 5. Suggested Future Cleanup

To improve readability and public usability, the code can be further reorganized into separate scripts such as:

- raw data preparation
- tokenizer / vectorizer training
- text feature construction
- model training and evaluation

This would make the pipeline easier to understand and maintain as a public portfolio project.
