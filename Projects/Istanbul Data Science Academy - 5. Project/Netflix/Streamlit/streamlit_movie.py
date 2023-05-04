import streamlit as st
import pandas as pd
import datetime


# Sayfa Ayarları
st.set_page_config(
    page_title="Recommendation",
    page_icon="https://www.citypng.com/public/uploads/preview/-11594687246vzsjesy7bd.png",
    menu_items={
        "Get help": "mailto:hasanenesguray@gmail.com",
        "About": "For More Information\n" + "https://github.com/hasanenesguray/IDSA-DSBootcamp"
    }
)

# Başlık Ekleme
st.title("Netflix - Movie Recommendation System")

# Markdown Oluşturma
st.markdown("A blog site called **:red[The Movie Blog]** , which shares movie content, reviews and comments with its followers, wants to increase the number of clicks, appeal to more users and thus increase advertising revenues. However, he realizes that the suggestions at the bottom of each blog post do not attract enough attention from the users. That's why they want to improve their recommendations.")


# Resim Ekleme
st.image("data:image/jpeg;base64,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")

st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a **recommmendation system** in line with their needs and help them with their research.")
st.markdown("In addition, when they have a title of a movie, they want us to come up with a recommendation list that icnludes 10 similar movies.")
st.markdown("*Let's help them!*")

st.image("https://1000logos.net/wp-content/uploads/2017/05/Netflix-Logo.png")

# Pandasla veri setini okuyalım ve düzenleyelim
df_title = pd.read_csv('df_title.csv')
df_title['Year'] = df_title['Year'].fillna(-1)
df_title['Year'] = df_title['Year'].astype('int')
df = pd.read_csv('df.csv')
df['Rating'] = df['Rating'].astype('int')
df_movie_summary = pd.read_csv('df_movie_summary.csv')

# Tablo Ekleme
st.header("Data Dictionary of Movie Ratings")
st.table(df.sample(5, random_state=42))

st.markdown("- **Cust_Id**: Unique customer identifier, who watched and rated the movie out of 5")
st.markdown("- **Rating**: Customer's rating out of 5")
st.markdown("- **Movie_Id**: Unique movie id, which customer watched and rated out of 5")

st.header("Data Dictionary of Movie Titles")
st.table(df_title.sample(5, random_state=42))

st.markdown("- **Year**: The year the movie was shot")
st.markdown("- **Name**: The title of the movie")


# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Year", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Name", help="Please capitalize the first letter of your surname!")
movie_input = st.sidebar.selectbox("Movie", tuple(df_title['Name'].unique()), help="Please fill the blank with a written review!")
# Model



def recommend(movie_title, min_count=0):
    df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
    i = int(df_title.index[df_title['Name'] == movie_title][0]) + 1
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
    result_df = corr_target[corr_target['count']>min_count][1:11]
    return result_df["Name"]

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):
    recommendations = recommend(movie_input,0)

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
    'Movie': [movie_input]
    })

    st.table(results_df)
    st.table(recommendations)
    st.image("https://uxwing.com/wp-content/themes/uxwing/download/video-photography-multimedia/film-movie-reel-icon.png")

else:
    st.markdown("Please click the *Submit Button*!")