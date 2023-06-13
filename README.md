# cs145-stock-prediction-project

This project combines two broad categories of stock analysis, Fundamental Analysis and Technical Analysis, to learn correlations between news reports and stock market action as a means to predict next-day stock price direction given prior day news. 

Assumptions: Good news correlates to upward market movement, and bad news correlates to downward market movement. 

To quantify good news and bad news we used sentiment analysis and for predictive ability we used a neural network (we assume the reader has foundational knowledge of both topics).

## Data
All data was procurred from Kaggle, an online community of data scientists and machine learning practitioners. 

[Stock Market Data](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) was used for daily stock time series data.

[Daily Financial News for 6000+ Stocks](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) was used for texutal headline sentiment analysis

[Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news?select=all-data.csv) was used to train our ensemble sentiment classifier.

The first two datasets were cleaned and combined to form a subset of 50 stocks and corresponding news covering a ten year timeframe.

`split_data-20230613T021632Z-001.zip` and `filtered_data-20230613T021725Z-001.zip` are similar datasets that differ in how data is grouped. Both datasets are required for generating classifier and network input vectors.

`headlines/sentiment_analysis_financial_news/all-data.csv` is used train sentiment classifiers.

## Code
This project has three main sections: Statistical Analysis, Sentiment Classification, Neural Network Prediction. Statistical analysis and Sentiment classification have no inter-dependencies and can therefore be run in any order. The neural network requires all sentiment classifiers to have been trained and a pickled ensemble output dataframe `filtered_data/majority_vote_sentiment.pkl`. 

Note: there are several cached objects in `pickles` and `filtered_data` that can be used instead of training new models.