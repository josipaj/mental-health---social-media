# mene zanima od subreddita ka naslov i tekst al uglavnom tekst bez naslova i autor

from pkgutil import get_data
from tkinter import N, X
from psaw import PushshiftAPI
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime as dt
import time
import threading
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

DEFAULT_DATA_SIZE = 10

def get_data(subreddit, start_epoch, name, num_of_data, df):
    api = PushshiftAPI()
    data = api.search_submissions(after=start_epoch,
                                subreddit=subreddit,
                                filter=['url','author', 'title', 'selftext', 'subreddit'],
                                limit=num_of_data)
    
    # ovo sve je za dohvatit podatke i ocistit i stavit u dataframe
    # uzeli 10 zadnjih objava(naslov i tekst) stavili u dataframe i uklonili nezeljene stupce
    
    ######################
    # uvik od podataka triba napravit dataframe i to je ovo dole
    subs = pd.DataFrame([submission.d_ for submission in data]).drop(labels=['url', 'created_utc', 'created'], axis=1)
    subs['text'] = subs['selftext'].astype(str) + '.' + subs['title']
    subs = subs.rename(columns={'subreddit': 'disorder'})
    subs['disorder'] = name
    new = clean(subs.drop(labels=['selftext', 'title'], axis =1)).reset_index(drop=True)
    


    # ako je df prazan onda se sve doli nastavlja normalno, a ako nije prazan samo ga dodaj u subs 
    if not df.empty:
        new = pd.concat([df, new], ignore_index=True)
    
    new = new.drop_duplicates()
    new = new.reset_index(drop = True)

    #new = edit_columns(new, name)
 
    # ostali stupci author, selftext i title, valjalo bi spojit selftext i title sa tipa tockon.
    # subs['text'] = subs['selftext'].astype(str) + '.' + subs['title']
    # subs = subs.rename(columns={'subreddit': 'disorder'})
    # subs['disorder'] = name
    # new = clean(subs.drop(labels=['selftext', 'title'], axis =1))
    print(new)
    data_len= len(new['disorder'] == name)
    #print(len(new['disorder'] == name)) #ovo je dobro ispisalo
    # moran smislit sta sad s ovin
    print(data_len)
    if data_len < DEFAULT_DATA_SIZE:
        new = get_data(subreddit, start_epoch, name, DEFAULT_DATA_SIZE+data_len, new)
    
    
    return new.iloc[:10]


def clean(data):
    # ukloni sve znakove, nove linije i tabove, i sve sta nije alfanumericko osin ' a mogu i brojeve isto to cu vidit jos
    data['text'] = data['text'].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r", "[^\w/']"], value=["",""," "], regex=True)
    #ukloni prazne tekstove, one koje sadrze manje od 3 rici i one koji pocinju na removed
    
    #ovo je da pribroji duzinu rici a ne broj rici
    # data['length'] = data['text'].str.len()
    # new = data[data['length'] > 3]

    #ovo je uklonilo sve retke di je broj rici manji od 3, znaci ukljucujuci i prazne
    data['totalwords'] = data['text'].str.split().str.len()
    emptyclean = data[(data['totalwords'] > 3) & (data['totalwords'] < 200)]

    # ovo radi al mi govori da prominin sa loc jer je bice ovo malo zastarjelo    
    emptyclean['firstword'] = emptyclean['text'].str.split().str[0]
    
    fullclean = emptyclean[emptyclean['firstword'] != 'removed']

    #da ukloni space-ove na pocetku i na kraju stringa
    fullclean['text'] = fullclean['text'].str.strip()

    # .drop(labels=['totalwords', 'firstword'], axis =1)
    return fullclean


def train(data):
    X = data['text']
    y = data['disorder']
    
    # pola podataka podili da bude test, pola train
    # x su znaci tekstovi, a y koja je bolest 
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5, shuffle=True)
    #print(y_train)

    ################################
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)
    
    #print(x_train_counts.shape)
    # ovo mi ne triba a iskreno i ne kuzin za sta se tocno koristi
    #print(f"vocab => {count_vect.vocabulary_.get(u'disaster')}")

    ################################
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    #print(X_train_tfidf.shape)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    docs_new = ['I want to kill myself', 'I love KFC']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print(f'{doc} -> {category}')

  


def main():
    st = time.time()
    start_epoch=int(dt.datetime(2020, 8, 24).timestamp())

    # bipolar disorder - BipolarReddit
    # depression - depression_help
    # OCD - OCD 
    # anxiety - Anxiety - Anxietyhelp
    # schizophrenia - schizophrenia - shizoaffective
    df = pd.DataFrame()

    bipolar_data = get_data('BipolarReddit', start_epoch, 'bipolar disorder', DEFAULT_DATA_SIZE, df)
    depression_data = get_data('depression_help', start_epoch, 'depression', DEFAULT_DATA_SIZE, df)
    ocd_data = get_data('OCD', start_epoch, 'OCD', DEFAULT_DATA_SIZE, df)
    schizophrenia_data = get_data('schizophrenia', start_epoch, 'schizophrenia', DEFAULT_DATA_SIZE, df)
    

    all_data = pd.concat([bipolar_data, depression_data, ocd_data, schizophrenia_data], ignore_index=True)
    all_data.to_csv('data.csv')

    #train(bipolar_data)

    et = time.time()
    print("Vrijeme: ", et-st)

    # #kod threds-a
    # st1 = time.time()
    
    # thread1 = threading.Thread(target=get_data, args=('BipolarReddit', start_epoch, 'bipolar disorder'))
    # thread2 = threading.Thread(target=get_data, args=('depression_help', start_epoch, 'depression'))
    # thread3 = threading.Thread(target=get_data, args=('OCD', start_epoch, 'OCD'))
    # thread4 = threading.Thread(target=get_data, args=('schizophrenia', start_epoch, 'schizophrenia'))
    # # Start the threads
    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()
    
    # # Join the threads before 
    # # moving further
    # thread1.join()
    # thread2.join()
    # thread3.join()
    # thread4.join()
    # bipolar_data1=thread1
    # depression_data1=thread2.value
    # ocd_data1=thread3.value
    # schizophrenia_data1=thread4.value

    # bipolar_data1.to_csv('bipolar_data1.csv')
    # depression_data1.to_csv('depression_data1.csv')
    # ocd_data1.to_csv('ocd_data1.csv')
    # schizophrenia_data1.to_csv('schizophrenia_data1.csv')

    # et1 = time.time()
    # print("Vrijeme threadingom: ", et1-st1)


if __name__ == "__main__":
    main()
    