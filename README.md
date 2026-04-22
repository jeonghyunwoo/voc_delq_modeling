# NLP-based Delinquency Resolution Prediction using VOC Text

Collection-call VOC text often contains early signals that are not fully captured by structured delinquency metrics alone.

This project uses counseling text from delinquent accounts to build an NLP-based prediction model for **delinquency resolution probability**, with the goal of supporting **earlier risk detection** and **operational prioritization** in retail credit management.

---

## 1. Business Context

In delinquency management, structured indicators such as delinquency days, balance, or bucket transitions are useful, but they may not fully reflect what is actually happening at the customer-contact level.

Collection-call notes often contain signals such as:

- willingness to repay
- uncertainty or avoidance
- temporary hardship
- repeated delays
- evidence of imminent resolution

These signals are embedded in unstructured text and are difficult to use consistently in day-to-day monitoring without a systematic NLP pipeline.

---

## 2. Objective

The purpose of this project is to estimate the probability that a delinquent account will be resolved using VOC text from collection calls.

The model is intended to support:

- earlier identification of worsening cases
- prioritization of collection workload
- more structured use of counselor notes
- risk monitoring beyond structured delinquency indicators alone

---

## 3. Project Idea

The overall workflow is:

1. collect and preprocess VOC text from collection calls
2. build a tokenizer suited to domain-specific expressions and slang
3. vectorize text using TF-IDF and Doc2Vec
4. aggregate text signals at the loan level
5. train a classification model to predict delinquency resolution probability

This project focuses not only on text modeling itself, but also on how unstructured operational data can be translated into a practical decision-support tool.

---

## 4. Why a Custom Tokenizer?

VOC text in collection environments often includes:

- abbreviations
- shorthand expressions
- inconsistent spacing
- domain-specific jargon
- counselor-specific writing styles

Because of this, a generic tokenizer may miss important patterns.

To better reflect actual call-note language, this project applies a **soynlp-based unsupervised tokenizer**, then uses the tokenized corpus to build downstream vectorization models.

---

## 5. Data and Target

The project uses delinquency counseling data combined with account-level portfolio information.

Examples of input fields include:

- observation month
- loan/account identifiers
- delinquency occurrence / resolution dates
- maturity information
- call type / counseling text
- customer / product / region-related attributes

The target variable is designed to indicate whether the delinquent case is resolved under the defined operational rule.

---

## 6. Methodology

### Step 1. Raw data preparation
Monthly delinquency counseling records are joined with portfolio and account data.

### Step 2. Text preprocessing
Records with no meaningful text are removed, and relevant counseling types are selected.

### Step 3. Tokenizer training
A domain-adapted tokenizer is trained using unlabeled VOC text with soynlp.

### Step 4. Text vectorization
Two vectorization approaches are used:

- **TF-IDF**
- **Doc2Vec**

### Step 5. Loan-level feature construction
When multiple call records exist for the same loan, text vectors are aggregated to create loan-level model input.

### Step 6. Prediction modeling
XGBoost-based classification is used to predict delinquency resolution probability for different management groups.

---

## 7. Repository Structure

```text
codes/
├─ 0_vectorizer_modeling.py   # tokenizer and Doc2Vec preparation
├─ 1_voc_modeling.py          # raw data prep, feature generation, modeling
└─ README.md
```

---

## 8. Practical Relevance

This project is not just a text-classification exercise.

Its practical value lies in showing how counselor VOC — which is often treated as difficult-to-use free text — can be turned into a structured probability signal for real operational use.

Potential use cases include:

- prioritizing collection targets
- supporting early-warning monitoring
- complementing structured delinquency indicators
- improving resource allocation in collection operations

---

## 9. Public Repository Notes

This repository is a simplified public version of an internal project.

Because the original work relied on internal portfolio data, database access, and environment-specific files, the public version does not include:

- source datasets
- internal DB connection logic
- private utility modules
- full production-ready execution environment

The repository is therefore intended to show:

- the business problem
- the NLP approach
- the feature engineering logic
- the operational use case

rather than to provide a fully reproducible public pipeline.

---

## 10. Tech Stack

- Python
- pandas
- soynlp
- gensim (Doc2Vec)
- scikit-learn
- XGBoost

---

## 11. Future Improvements

- add synthetic sample VOC data
- separate preprocessing / vectorization / modeling into modular scripts
- clean environment-specific paths and imports
- document evaluation results and threshold strategy
- include example outputs for operational prioritization
