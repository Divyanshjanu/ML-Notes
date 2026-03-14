# 🎬 Naive Bayes — Movie Review Sentiment Analysis

A beginner-friendly project that uses **Naive Bayes Classifier** to predict whether an IMDB movie review is **Positive** or **Negative**.

**Dataset:** [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (25,000 positive + 25,000 negative reviews)

---

## 📖 What is Naive Bayes?

Naive Bayes is a **probabilistic classifier** — it uses probability to decide which class (positive or negative) a review belongs to. It is based on **Bayes' Theorem**, a formula from probability theory.

Think of it like this: you read a movie review that contains words like *"amazing"*, *"brilliant"*, *"loved"*. Your brain automatically thinks — this is probably a positive review. Naive Bayes does the exact same thing, but with math.

---

## 🧮 The Math Behind Naive Bayes

### Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

In simple words: **"What is the probability of event A happening, given that event B has already happened?"**

### Applying it to Sentiment Analysis

Let's say we want to find: **What is the probability that a review is Positive, given the words in it?**

$$P(\text{Positive} \mid \text{words}) = \frac{P(\text{words} \mid \text{Positive}) \times P(\text{Positive})}{P(\text{words})}$$

Let's break down each term:

| Term | Name | Meaning |
|------|------|---------|
| P(Positive \| words) | **Posterior** | What we want to find — probability of positive given the words |
| P(words \| Positive) | **Likelihood** | How often do these words appear in positive reviews? |
| P(Positive) | **Prior** | What % of all reviews are positive? (In our case ~50%) |
| P(words) | **Evidence** | How common are these words across all reviews? |

### A Simple Example

Imagine we have a tiny dataset of 10 reviews:
- **6 Positive** reviews
- **4 Negative** reviews

And we know that:
- The word **"amazing"** appears in **5 out of 6** positive reviews
- The word **"amazing"** appears in **1 out of 4** negative reviews
- The word **"amazing"** appears in **6 out of 10** total reviews

Now a new review comes in: **"This movie is amazing"**

**Step 1: Calculate Prior Probabilities**
```
P(Positive) = 6/10 = 0.6
P(Negative) = 4/10 = 0.4
```

**Step 2: Calculate Likelihoods**
```
P("amazing" | Positive) = 5/6 = 0.833
P("amazing" | Negative) = 1/4 = 0.25
```

**Step 3: Calculate Evidence**
```
P("amazing") = 6/10 = 0.6
```

**Step 4: Apply Bayes' Theorem**
```
P(Positive | "amazing") = (0.833 × 0.6) / 0.6 = 0.833
P(Negative | "amazing") = (0.25 × 0.4) / 0.6  = 0.167
```

**Result:** 83.3% chance of Positive vs 16.7% chance of Negative → **Classified as Positive! ✅**

### Why is it called "Naive"?

The **"Naive"** part comes from a simplifying assumption: it assumes all words are **independent** of each other.

For example, if a review contains both "not" and "good", Naive Bayes treats them as separate words. It doesn't understand that "not good" together means something negative. Each word is evaluated on its own.

This assumption is obviously not true in real language — but surprisingly, Naive Bayes still works very well in practice!

### What is Multinomial Naive Bayes?

There are different types of Naive Bayes depending on the kind of data:

| Type | When to Use | Example |
|------|-------------|---------|
| **Multinomial NB** | Word counts / frequencies | Text classification (our case) |
| **Gaussian NB** | Continuous numbers | Height, weight, temperature |
| **Bernoulli NB** | Yes/No (binary) features | Word present or absent |

We use **Multinomial NB** because our features are word counts (how many times each word appears in a review).

### What is Laplace Smoothing (alpha)?

What if a word in the test review **never appeared** in training data? Then:

```
P(new_word | Positive) = 0/1000 = 0
```

This would make the **entire probability zero** — which is bad! One unseen word would break everything.

**Laplace Smoothing** fixes this by adding a small number (alpha, usually 1) to every word count:

```
P(new_word | Positive) = (0 + 1) / (1000 + vocab_size)
```

Now it's a very small number instead of zero. Problem solved!

---

## 🔧 How the Code Works

### Pipeline Overview

```
Raw Text → Clean Text → Bag of Words (Numbers) → Naive Bayes → Prediction
```

**Step 1: Text Cleaning**
- Remove HTML tags (`<br/>` etc.)
- Remove special characters and numbers
- Convert to lowercase

**Step 2: Bag of Words (CountVectorizer)**
- Converts text into numbers by counting word frequencies
- Example:
  ```
  "good movie"      → [0, 1, 1]
  "bad movie bad"   → [2, 0, 1]
                       bad good movie
  ```

**Step 3: Train/Test Split**
- 80% data for training, 20% for testing

**Step 4: Multinomial Naive Bayes**
- Learns word probabilities from training data
- Uses Bayes' Theorem to classify new reviews

**Step 5: Evaluation**
- Accuracy, Confusion Matrix, Classification Report

---

## 📊 Results

| Metric | Negative | Positive |
|--------|----------|----------|
| **Precision** | 0.84 | 0.85 |
| **Recall** | 0.85 | 0.84 |
| **F1-Score** | 0.85 | 0.85 |
| **Support** | 4961 | 5039 |

| | Score |
|--|-------|
| **Accuracy** | **84.58%** |
| **Macro Avg F1** | 0.85 |
| **Weighted Avg F1** | 0.85 |

Tested on **10,000 reviews** — solid result for such a simple model with minimal preprocessing!

---

## 📁 Project Structure

```
Naive_Bayes/
├── README.md
├── Naive_bayes_sentiment_analysis.ipynb
└── IMDB Dataset.csv ( Download it from Kaggle )
```

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**2. Download dataset:**

Download from [Kaggle - IMDB 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and save as `IMDB Dataset.csv` in the same folder.

**3. Run the notebook:**
```bash
jupyter notebook Naive_bayes_sentiment_analysis.ipynb
```

Run all cells top to bottom.

---

## 🔮 Possible Improvements

- Use **TF-IDF** instead of Bag of Words — gives more weight to important words
- Add **bigrams** (2-word pairs like "not good") to capture context
- Apply **stemming/lemmatization** — treat "running", "runs", "ran" as the same word
- Tune the **alpha** parameter to find the best smoothing value
- Compare with other models like **Logistic Regression** or **SVM**

---

## 📚 References

- [Naive Bayes — scikit-learn Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [IMDB Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Bayes' Theorem — Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)

---

## 👤 Author

**Divyanshjanu** — [GitHub](https://github.com/Divyanshjanu)
