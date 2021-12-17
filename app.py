import streamlit as st
import joblib
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

reg_token=RegexpTokenizer('[a-zA-Z]+')
sw=set(stopwords.words('english'))
wnl=WordNetLemmatizer()

def simple_pos(p):
  if p.startswith('J'):
    return wordnet.ADJ
  elif p.startswith('V'):
    return wordnet.VERB
  elif p.startswith('N'):
    return wordnet.NOUN
  elif p.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN 

def clean_data(k):
  d=reg_token.tokenize(k)
  cleaned_words=[]
  for w in d:
    if w.lower() not in sw:
      p=pos_tag([w])
      word=wnl.lemmatize(w,pos=simple_pos(p[0][1]))
      cleaned_words.append(word.lower())

  return " ".join(cleaned_words)

def load():
  model=joblib.load('D:/projects/Flickorflop sentiment analysis/model.sav')
  vectorizer=joblib.load('D:/projects/Flickorflop sentiment analysis/vectorizer.sav')
  return model,vectorizer

model,vectorizer=load()

def predict_sentiment(y):
  
  y_cleaned=clean_data(y)
  if(y_cleaned==''):
    result='Invalid Input'
  else:
    result=model.predict(vectorizer.transform([y_cleaned]))
  st.success(result[0])


st.title('FlickOrFlop')
st.text('Sentiment Analysis on IMDB dataset')
st.subheader('Prediction on User input')
input=st.text_input('Enter movie review ',' ')
if st.button('Display Sentiment'):
  predict_sentiment(input)

