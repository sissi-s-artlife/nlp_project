import nltk
import ssl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download VADER lexicon if not already downloaded
nltk.download("vader_lexicon", quiet=True)

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Input text for sentiment analysis
text = input("Type the text you want to check the sentiment score of: ")

# Get sentiment scores
sentiment_scores = sia.polarity_scores(text)

# Interpret the sentiment scores
compound_score = sentiment_scores["compound"]

if compound_score >= 0.05:
    sentiment = "positive"
elif compound_score <= -0.05:
    sentiment = "negative"
else:
    sentiment = "neutral"

# Print the sentiment and sentiment scores
print("Sentiment:", sentiment)
print("Sentiment Scores:", sentiment_scores)
