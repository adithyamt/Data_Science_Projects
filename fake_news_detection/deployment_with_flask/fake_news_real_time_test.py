import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from flask import Flask,render_template,request,url_for

app=Flask(__name__)

@app.route('/')
def news():
    return render_template('news.html')

def news_predict(to_predict_news):
    model = keras.models.load_model('news.h5')
    nltk.download('stopwords')
    stop_word = set(stopwords.words('english'))
    max_features = 5000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(to_predict_news)
    filtered_word = []
    for word in to_predict_news:
        if word in stop_word:
           filtered_word.append(word)
    seq1 = tokenizer.texts_to_sequences(filtered_word)
    seq = pad_sequences(seq1, maxlen=9615)
    pred = model.predict_classes(seq)
    return pred[0]

@app.route('/result',methods=['POST','GET'])
def result():
    if request.method=='POST':
        to_predict_news=request.get_data(as_text=True)
        result=news_predict(to_predict_news)

        if result == 1:
            prediction='Real News '
        else:
            prediction='Fake News'

        return render_template('result.html',prediction=prediction)


if __name__=='__main__':
    app.run(debug=True)

