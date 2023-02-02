import streamlit as st
import psycopg2
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import re
import string
from joblib import load
from nltk.stem import SnowballStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer



# Sayfa Ayarları
st.set_page_config(
    page_title="Amazon - iPhone 13 Reviews: Sentiment Analysis",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/d/de/Amazon_icon.png",
    menu_items={
        "Get help": "mailto:hasanenesguray@gmail.com",
        "About": "For More Information\n" + "https://github.com/hasanenesguray/IDSA-DSBootcamp"
    }
)

# Başlık Ekleme
st.title("Amazon - iPhone 13 Reviews: Sentiment Analysis")

# Markdown Oluşturma
st.markdown("**:orange[Amazon]** wants to analyze the sentiment analysis of customer reviews.")

# Resim Ekleme
st.image("https://pngimg.com/uploads/iphone_13/iphone_13_PNG27.png")

st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a **machine learning model** in line with their needs and help them with their research.")
st.markdown("In addition, when they have information about a song feature, they want us to come up with a reviews list that we can predict whether one of the reviews include positive or negative feelings.")
st.markdown("*Let's help them!*")

st.image("https://wildfiresocial.com/wp-content/uploads/2019/01/amazon-logo-white._cb1509666198_.png")

# Pandasla veri setini okuyalım ve düzenleyelim
df = pd.read_csv('amazon_final.csv')
df.drop(['Unnamed: 0'], axis=1,inplace=True)

# Tablo Ekleme
st.table(df.sample(5, random_state=42))


# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **productCode**: Unique Amazon identifier for the product that this row is a part of")
st.markdown("- **star**: Criterion that the customer gives to the relevant product out of 5 and shows how much s/he likes the product")
st.markdown("- **review**: Customer's written comment on the relevant product that can be viewed by everyone")

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
review_input = st.sidebar.text_input("Review", help="Please fill the blank with a written review!")


# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması


logreg_model = load('amazon_comments_model.pkl')

# review_input = 'I like it, battery long life, the phone so good.'

input_df = pd.DataFrame({
     'review': [review_input]
})


# Removing the \n expressions from the reviews
input_df['review'] = input_df['review'].str.replace("\n","")
# Removing the numbers from the reviews
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
# Removing the punctations from the reviews ve converting all letters to lowercase
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) 

input_df['review'] = input_df.review.map(alphanumeric).map(punc_lower)

sbs = SnowballStemmer(language='english')
def stemmer(text):
    text = [sbs.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

input_df.review = input_df.review.apply(stemmer)


def correct(text):
    text = [str(TextBlob(word).correct()) for word in text.split(' ')]
    text = " ".join(text)
    return text

input_df.review = input_df.review.apply(correct)

X = input_df.review

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):
    # Tokenizing using the 2-Way N-Gram method
    cv2 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')

    vectorized_input = cv2.fit_transform(X)
    vectorized_input_df = pd.DataFrame(vectorized_input.toarray(), columns=cv2.get_feature_names_out()).head()
    vectorized_input_dict = vectorized_input_df.to_dict()

    as_is_df = pd.read_csv('result_table.csv')

    column_headers = list(as_is_df.columns.values)

    out = dict.fromkeys(column_headers, 0)

    for i in list(vectorized_input_df.columns.values):
        if i in column_headers:
            val = vectorized_input_dict[i][0]
            out[i]= val


    out_df = pd.DataFrame([out])
    out_df = out_df.drop("Unnamed: 0", axis='columns')

    pred = logreg_model.predict(out_df)
    pred_probability = np.round(logreg_model.predict_proba(out_df), 2)

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'review': [review_input],
    'Prediction': pred,
    'Negative Probability': [pred_probability[:,:1]],
    'Positive Probability': [pred_probability[:,1:]]
    })

    st.table(results_df)

    if pred == ['Positive']:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Smiley.svg/1200px-Smiley.svg.png")
    else:
        st.image("https://em-content.zobj.net/source/noto-emoji-animations/344/pensive-face_1f614.gif")
else:
    st.markdown("Please click the *Submit Button*!")
