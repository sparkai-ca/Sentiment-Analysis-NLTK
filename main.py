import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize



def get_word_sentiment(text, sentiment):
    tokenized_text = nltk.word_tokenize(text)

    pos_word_list = []
    neu_word_list = []
    neg_word_list = []
    confidence = 0

    for word in tokenized_text:
        if (analyzer.polarity_scores(word)['compound']) >= 0.1:
            confidence = analyzer.polarity_scores(word)['pos']
            pos_word_list.append(word)

        elif (analyzer.polarity_scores(word)['compound']) <= -0.1:
            confidence = analyzer.polarity_scores(word)['neg']
            neg_word_list.append(word)

        else:
            confidence = analyzer.polarity_scores(word)['neu']
            neu_word_list.append(word)

    if sentiment == 'positive':
        return " ".join(pos_word_list), confidence



    elif sentiment == 'negative':
        return " ".join(neg_word_list), confidence

    else:
        return " ".join(neu_word_list), confidence


df_test = pd.read_csv('PATH_TO_TEST_CSV')
df_test = df_test.astype(str)

words_list = []
confidence_list = []

for i, r in df_test.iterrows():
    words, confidence = get_word_sentiment(r[1], r[2])

    words_list.append(words)
    confidence_list.append(confidence)

word_column = pd.DataFrame(words_list, columns=['selected_text'])
confidence_column = pd.DataFrame(confidence_list, columns=['confidence'])


submission = pd.concat([df_test, word_column,confidence_column], ignore_index=True,axis=1)
submission = submission.drop([1,2],axis=1)
submission.columns = ['textID','selected_text','confidence']
submission.to_csv("PATH_TO_SAVE")