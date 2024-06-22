# Topic and Sentiment Classification of News Articles

## Overview
This project focuses on classifying news articles into specific market topics and analyzing their sentiment. The aim is to help investors and companies understand market trends and public sentiment, which are crucial for making informed financial decisions.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Vectorization](#vectorization)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [Authors](#authors)
- [Usage](#usage)
- [License](#license)

## Introduction
Financial companies and investors rely heavily on stock analysis to make informed decisions. News articles play a significant role in shaping these decisions by providing insights into market events, company performances, and economic changes. This project aims to classify news articles into five market topics: Agriculture, Housing, Stocks, Manufacturing, and Technology. Additionally, it predicts the sentiment (Positive, Negative, Neutral) of these articles.

## Data Collection
### News Articles Labeled by Topic
- **Source**: News API (newsapi.org)
- **Topics**: Agriculture, Housing, Stocks, Manufacturing, Technology
- **Collection Period**: 8/26/2023 to 9/14/2023
- **Total Articles**: 9328 after cleaning and removing duplicates

### News Articles Labeled by Sentiment
- **Source**: Kaggle (Financial Sentiment Analysis dataset)
- **Sentiment Labels**: Positive, Negative, Neutral
- **Total Sentences**: 5836 after cleaning and removing duplicates

## Libraries Used
- `pandas`: For data manipulation and organization.
- `numpy`: For numerical operations.
- `sklearn`: For machine learning models and vectorization.
- `matplotlib.pyplot`, `seaborn`, `wordcloud`: For data visualization.
- `newsapi`: To fetch news articles from News API.

## Data Preprocessing
### Cleaning and Preparation
- **Duplicates Removal**: Eliminated duplicate rows and rows with missing data.
- **Balancing Data**: Created a balanced dataset for sentiment analysis to avoid bias.

## Vectorization
### Topic Analysis
- **Vectorizer**: CountVectorizer
- **Parameters**:
  - Lowercase conversion
  - Maximum features set to 1500
  - Removed English stopwords and specific topic words
- **Word Clouds**: Generated to visualize common words in overall and topic-specific articles.

### Sentiment Analysis
- **Vectorizer**: TfidfVectorizer
- **Parameters**:
  - Lowercase conversion
  - Maximum features set to 1500
  - Removed English stopwords
- **Word Clouds**: Generated to visualize common words in overall and sentiment-specific articles.

## Modeling
### Algorithms Used
1. **Naive Bayes**:
   - Assumes independence among words.
   - Uses word frequencies for classification.
   - Implemented with `MultinomialNB` from Scikit-Learn.
2. **Support Vector Machine (SVM)**:
   - Finds the optimal hyperplane for classification.
   - Uses linear kernel for text data.
   - Implemented with `SVC` from Scikit-Learn.
3. **Decision Tree**:
   - Non-linear model that splits data based on feature values.
   - Visualizes decision-making process.
   - Implemented with `DecisionTreeClassifier` from Scikit-Learn.
4. **Latent Dirichlet Allocation (LDA)**:
   - Identifies hidden topics in text.
   - Assumes documents are mixtures of topics and topics are mixtures of words.
   - Implemented with `LatentDirichletAllocation` from Scikit-Learn.

### Model Tuning and Evaluation
- **Cross-Validation**: 3-fold cross-validation for performance evaluation.
- **Confusion Matrix**: To visualize model predictions.
- **Accuracy Metrics**: Used to compare model performance.

## Results
### Topic Classification
- **Best Model**: Naive Bayes with Count Vectorization
  - **Accuracy**: Highest cross-validation and test accuracy.
  - **Confusion Matrix**: Stocks had the highest precision and recall.
  - **Important Words**: Key words influencing topic predictions were identified.

### Sentiment Classification
- **Best Model**: Support Vector Machine with Tfidf Vectorization on Unbalanced Data
  - **Accuracy**: Best performance on unbalanced data.
  - **Confusion Matrix**: High precision and recall for neutral sentiment.
  - **Balanced Data Performance**: Improved negative sentiment prediction.

## Conclusion
This analysis successfully created models to classify news articles by topic and predict their sentiment. The models can be integrated into larger systems to provide valuable insights for financial analysis and decision-making.

## Authors
- **Evan Helig** - eahelig@syr.edu
- **Dimetrius Johnson** - djohns66@syr.edu
- **Sean Deery** - sdeery@syr.edu

## Usage
To run this project locally:
1. Clone the repository: `git clone <repository-url>`
2. Install required libraries
3. Run the python files

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
