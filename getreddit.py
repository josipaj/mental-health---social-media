from psaw import PushshiftAPI
import pandas as pd

pd.options.mode.chained_assignment = None
import datetime as dt
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


DEFAULT_DATA_SIZE = 10000

def get_data(subreddit, start_epoch, name, num_of_data, wanted_num):
    api = PushshiftAPI()
    data = api.search_submissions(after=start_epoch,
                                subreddit=subreddit,
                                filter=['url','author', 'title', 'selftext', 'subreddit'],
                                limit=num_of_data)
    
    subs = pd.DataFrame([submission.d_ for submission in data]).drop(labels=['url', 'created_utc', 'created'], axis=1)
    subs['text'] = subs['selftext'].astype(str) + '.' + subs['title']
    subs = subs.rename(columns={'subreddit': 'disorder'})
    subs['disorder'] = name
    new = clean(subs.drop(labels=['selftext', 'title'], axis =1)).reset_index(drop=True)

    return new.iloc[:wanted_num]


def clean(data):
    # remove all new lines, tabs and non-alphanumerical signs instead of '
    data['text'] = data['text'].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r", "[^\w/']"], value=["",""," "], regex=True)
    
    # remove rows where length of word is less then 3 and more than 600
    data['totalwords'] = data['text'].str.split().str.len()
    emptyclean = data[(data['totalwords'] > 3) & (data['totalwords'] < 600)]
   
    emptyclean['firstword'] = emptyclean['text'].str.split().str[0]
    fullclean = emptyclean[emptyclean['firstword'] != 'removed']

    # remove spaces at the start and end of the string
    fullclean['text'] = fullclean['text'].str.strip()

    # remove authors that are deleted
    all =fullclean[fullclean['author'] != '[deleted]']

    all = all.drop_duplicates()
    all = all.reset_index(drop = True)

    return all.drop(labels=['totalwords', 'firstword'], axis =1)


def remove_contractions(text):
    import contractions
    expanded_words = []    
    for word in text.split():
        expanded_words.append(contractions.fix(word))   
        
    expanded_text = ' '.join(expanded_words)
    return expanded_text


def extra_clean(text):
    rm_contract = remove_contractions(text)
    
    tokens = nltk.word_tokenize(rm_contract)
  
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)


def print_confusion_matrix(model,
                           confusion_matrix,
                           fontsize=12,
                           ylabel='True label',
                           xlabel='Predicted label'):

    class_names = model.classes_
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    
    # heathmap1 is with numbers of data
    heatmap1 = sns.heatmap(df_cm, annot=True, fmt="d")
        
    heatmap1.yaxis.set_ticklabels(heatmap1.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap1.xaxis.set_ticklabels(heatmap1.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

    # heathmap2 is with percentages
    heatmap2 = sns.heatmap(df_cm/np.sum(df_cm), annot=True, fmt='.2%', cmap='Blues')

    heatmap2.yaxis.set_ticklabels(heatmap2.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap2.xaxis.set_ticklabels(heatmap2.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


def train(data):

    X = data['text']
    y = data['disorder']
    # y = data['class']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5, shuffle=True)

    ################################
    # call each function individually
    vectorizer = CountVectorizer()
    x_train_counts = vectorizer.fit_transform(x_train)

    tfidf_transform = TfidfTransformer()
    X_train_tfidf = tfidf_transform.fit_transform(x_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    
    # for test data
    X_new = vectorizer.transform(x_test)
    X_new_tfidf = tfidf_transform.transform(X_new)
    predicted = clf.predict(X_new_tfidf)

    df = pd.DataFrame()
    df['text'] = x_test
    df['disorder'] = y_test
    df['predicted'] = predicted
    df = df.reset_index(drop = True)
    df.to_csv('prediction.csv')

    ################################
    # calling all functions using pipeline
    with_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    with_pipeline.fit(x_train, y_train)

    print("Train size: 50%, test size: 50%")

    print(f"Accuracy:-> {with_pipeline.score(x_test,y_test)}")

    predicted = with_pipeline.predict(x_test)
    new = pd.DataFrame()
    new['text'] = x_test
    new['disorder'] = y_test
    new['predicted'] = predicted
    new = new.reset_index(drop = True)
    new.to_csv('pipeline.csv')

    # confusion matrix
    print(metrics.classification_report(y_test, predicted))
    cm = metrics.confusion_matrix(y_test, predicted, labels=with_pipeline.classes_)
    print_confusion_matrix(with_pipeline,confusion_matrix=cm)


def main():
    st = time.time()

    ################################
    # get all data and save them to .csv file
    # start_time=int(dt.datetime(2012, 1, 1).timestamp())
    # wanted_num_data = 5000

    # bipolar_data = get_data('BipolarReddit', start_time, 'bipolar disorder', DEFAULT_DATA_SIZE, wanted_num_data)
    # depression_data = get_data('depression_help', start_time, 'depression', DEFAULT_DATA_SIZE, wanted_num_data)
    # ocd_data = get_data('OCD', start_time,'OCD', DEFAULT_DATA_SIZE, wanted_num_data)
    # schizophrenia_data = get_data('schizophrenia', start_time,'schizophrenia', DEFAULT_DATA_SIZE, wanted_num_data)

    ################################
    # take some random subreddits 
    # data = get_data('lifehacks', start_time, 'test', DEFAULT_DATA_SIZE*2, wanted_num_data*2)
    # data2 = get_data('funny', start_time, 'test', DEFAULT_DATA_SIZE*2, wanted_num_data*2)
    # regular_data = pd.concat([data, data2], ignore_index=True)

    # all_data = pd.concat([bipolar_data, depression_data, ocd_data, schizophrenia_data, regular_data], ignore_index=True)
    # all_data = all_data.sample(frac = 1)
    # all_data.to_csv('newdata2.csv')

    ################################
    # get number of all data
    # bipolar= len(all_data[all_data['disorder'] == 'bipolar disorder'])
    # print("bipolar data number: ", bipolar)
    # depression = len(all_data[all_data['disorder'] == 'depression'])
    # print("depression data number: ", depression)
    # schizophrenia = len(all_data[all_data['disorder'] == 'schizophrenia'])
    # print("schizophrenia data number: ", schizophrenia)
    # ocd = len(all_data[all_data['disorder'] == 'OCD'])
    # print("OCD data number: ", ocd)
    # regular= len(all_data[all_data['disorder'] == 'test'])
    # print("regular data number: ", regular)

    all_data = pd.read_csv('newdata.csv')

    new_data = pd.DataFrame()
    new_data['clean text'] = ['text'].map(lambda x: extra_clean(x))
    new_data['disorder'] = all_data['disorder']
    new_data.to_csv('allclean.csv')
    
    # get suicide data
    suicide_data = pd.read_csv('Suicide_Detection.csv')
    print("Number of suicide label data: ", len(suicide_data[suicide_data['class'] == 'suicide']))
    print("Number of non-suicide label data: ", len(suicide_data[suicide_data['class'] == 'non-suicide']))

    train(all_data)
    train(suicide_data)

    # take a part of data and do the training for them
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy = {'test': 100000, 'depression': 25000, 
    'bipolar disorder': 25000, 'OCD': 25000, 'schizophrenia': 25000}, random_state=42)
    X_under, y_under = undersample.fit_resample(all_data[['text']], all_data['disorder'])
    equal_data_num = pd.DataFrame({'text':X_under['text'], 'disorder':y_under})
    equal_data_num = equal_data_num.sample(frac = 1)
    equal_data_num = equal_data_num.reset_index(drop = True)
    train(equal_data_num)
    
    equal_data_num.to_csv('equal_data_num.csv')

    et = time.time()
    print("Time: ", et-st)


if __name__ == "__main__":
    main()
    