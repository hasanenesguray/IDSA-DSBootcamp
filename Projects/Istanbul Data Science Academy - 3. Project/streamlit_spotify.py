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



# Sayfa Ayarları
st.set_page_config(
    page_title="Spotify Song Skip Classification",
    page_icon="https://www.iconlogovector.com/uploads/images/2022/01/spotify-vertical.png",
    menu_items={
        "Get help": "mailto:hasanenesguray@gmail.com",
        "About": "For More Information\n" + "https://github.com/hasanenesguray/IDSA-DSBootcamp"
    }
)

# Başlık Ekleme
st.title("Spotify Song Skip Classification")

# Markdown Oluşturma
st.markdown("**:green[Spotify]** wants to analyze which features of the songs determine whether the songs are skipped by the users.")

# Resim Ekleme
st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png")

st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a **machine learning model** in line with their needs and help them with their research.")
st.markdown("In addition, when they have information about a song feature, they want us to come up with a product that we can predict whether this song will be skipped bu the users.")
st.markdown("*Let's help them!*")

st.image("https://i0.wp.com/blog.happyfox.com/wp-content/uploads/2014/11/How-to-Rock-Customer-Engagement-like-Spotify.png?resize=1200%2C630&ssl=1")

# Pandasla veri setini okuyalım ve düzenleyelim
params = params = {
    "host": "localhost",
    "user": "postgres",
    "port": 5432
}
connection = psycopg2.connect(**params, dbname= "postgres")

df_log = pd.read_sql("select * from public.spotify_log;", connection)

df_track = pd.read_sql("select * from public.spotify_track;", connection)

df_log.drop('skip_1', inplace=True, axis=1)
df_log.drop('skip_3', inplace=True, axis=1)
df_log.drop('not_skipped', inplace=True, axis=1)

df_log["skip_2"] = df_log["skip_2"].astype(int)
df_log["hist_user_behavior_is_shuffle"] = df_log["hist_user_behavior_is_shuffle"].astype(int)
df_log["premium"] = df_log["premium"].astype(int)


# Header Ekleme
st.header("Data Dictionary")

st.markdown("The schema for the session logs is given below. Each row corresponds to the playback of one track, and has the following fields, with corresponding values")

st.markdown("- **session_id**: Unique identifier for the session that this row is a part of")
st.markdown("- **session_position**: Position of row within session")
st.markdown("- **session_length**: Number of rows in session")
st.markdown("- **track_id_clean**: Unique identifier for the track played. This is linked with track_id in the track features and metadata table.")
st.markdown("- **skip_2**: Boolean indicating if the track was only played briefly(0: not skipped, 1: skipped")
st.markdown("- **context_switch**: Boolean indicating if the user changed context between the previous row and the current row. This could for example occur if the user switched from one playlist to another.")
st.markdown("- **no_pause_before_play**: Boolean indicating if there was no pause between playback of the previous track and this track")
st.markdown("- **short_pause_before_play**: Boolean indicating if there was a short pause between playback of the previous track and this track")
st.markdown("- **long_pause_before_play**: Boolean indicating if there was a long pause between playback of the previous track and this track")
st.markdown("- **hist_user_behavior_n_seekfwd**: Number of times the user did a seek forward within track")
st.markdown("- **hist_user_behavior_n_seekback**: Number of times the user did a seek back within track")
st.markdown("- **hist_user_behavior_is_shuffle**: Boolean indicating if the user encountered this track while shuffle mode was activated")
st.markdown("- **hour_of_day**: The hour of day of the session")
st.markdown("- **date**: The date of the session")
st.markdown("- **premium**: Boolean indicating if the user was on premium or not. This has potential implications for skipping behavior.")
st.markdown("- **context_type**: What type of context the playback occurred within")
st.markdown("- **hist_user_behavior_reason_start**: The user action which led to the current track being played")
st.markdown("- **hist_user_behavior_reason_end**: The user action which led to the current track playback ending")

# Tablo Ekleme
st.table(df_log.sample(5, random_state=42))

st.markdown("The schema for the track metadata and features is given below, each row has the following fields, with corresponding values")

st.markdown("- **track_id**: Unique identifier for the track played. This is linked with track_id_clean in the session logs")
st.markdown("- **duration**: Length of track in seconds")
st.markdown("- **release_year**: Estimate of year the track was released")
st.markdown("- **us_popularity_estimate**: Estimate of the US popularity percentile of")
st.markdown("- **acousticness**: See https://developer.spotify.com/documentation/ web-api/reference/tracks/get-audio-features/")
st.markdown("- **acoustic_vector_0**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_1**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_2**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_3**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_4**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_5**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_6**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")
st.markdown("- **acoustic_vector_7**: See ​http://benanne.github.io/2014/08/05/spotify-cnns.html​ and http://papers.nips.cc/paper/5004-deep-content-based-")

# Tablo Ekleme
st.table(df_track.sample(5, random_state=42))

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
session_position = st.sidebar.number_input("Session Position", min_value=1, max_value=20, format="%d")
session_length = st.sidebar.number_input("Session Length", min_value=10, max_value=20, format="%d")
if st.sidebar.checkbox('Context Switch') == True:
    context_switch = 1
else:
    context_switch = 0
if st.sidebar.checkbox('Pause Before Play') == True:
    no_pause_before_play = 0
else:
    no_pause_before_play = 1

#Model

df = pd.read_sql("select * from public.spotify_log sl join public.spotify_track st on sl.track_id_clean = st.track_id;", connection)
def weekday_converter(date):
    format = '%Y-%m-%d'
    datetime_date = datetime.datetime.strptime(date,format)
    weekdays_list = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return weekdays_list[datetime_date.weekday()]
df = df.assign(weekday=df.date.apply(weekday_converter))
def track_age_calculator(year):
    return 2022-year
df = df.assign(track_age=df.release_year.apply(track_age_calculator))
df.drop('session_id', inplace=True, axis=1)
df.drop('track_id_clean', inplace=True, axis=1)
df.drop('skip_1', inplace=True, axis=1)
df.drop('skip_3', inplace=True, axis=1)
df.drop('not_skipped', inplace=True, axis=1)
df.drop('date', inplace=True, axis=1)
df.drop('track_id', inplace=True, axis=1)
df = pd.get_dummies(df, columns=['context_type', 'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end',
                                      'weekday','mode'], drop_first=True)
df["skip_2"] = df["skip_2"].astype(int)
df["hist_user_behavior_is_shuffle"] = df["hist_user_behavior_is_shuffle"].astype(int)
df["premium"] = df["premium"].astype(int)
columns = df.columns.tolist()
columns = [columns[2]] + columns[0:2] + columns[3:]
#df = df[columns]
df = df[['skip_2','session_position','session_length','context_switch','no_pause_before_play']]
X_train, x_test, Y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], 
                                                    test_size = 0.25, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state=42)
std_scale = StandardScaler()
X_train_scaled = std_scale.fit_transform(X_train)
x_train_scaled = std_scale.fit_transform(x_train)
x_val_scaled = std_scale.transform(x_val)
x_test_scaled = std_scale.transform(x_test)
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x_train_scaled, y_train)
logreg_model = logreg
input_df = pd.DataFrame({
    'session_position': [session_position],
    'session_length': [session_length],
    'context_switch': [context_switch],
    'no_pause_before_play': [no_pause_before_play],
})
input_df_scaled = std_scale.transform(input_df)
pred = logreg_model.predict(input_df_scaled)
pred_probability = np.round(logreg_model.predict_proba(input_df_scaled), 2)

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

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
    'session_position': [session_position],
    'session_length': [session_length],
    'context_switch': [context_switch],
    'no_pause_before_play': [no_pause_before_play],
    'Prediction': [pred],
    'Not Skip Probability': [pred_probability[:,:1]],
    'Skip Probability': [pred_probability[:,1:]]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Not Skip"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Skip"))

    st.table(results_df)

    if pred == 0:
        st.image("https://upload.wikimedia.org/wikipedia/en/2/26/Don%27t_Stop_the_Music_Single.PNG")
    else:
        st.image("https://lh3.googleusercontent.com/EFxkc8haQIViXYuLwew_zxsuhbI4jd8CMIyz-KHMCIH1Yjb__Hz5PsahLdc12w4T0QZLpZj5Bi42o_wVGUlIJaoF=w640-h400-e365-rj-sc0x00ffffff")
else:
    st.markdown("Please click the *Submit Button*!")
