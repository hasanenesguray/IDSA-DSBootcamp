import numpy as np
import pandas as pd
import streamlit as st

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

# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **session_id**: Unique identifier for the session that this row is a part of")
st.markdown("- **session_position**: Position of row within session")
st.markdown("- **session_length**: Number of rows in session")
st.markdown("- **track_id_clean**: Unique identifier for the track played. This is linked with track_id in the track features and metadata table.")
st.markdown("- **skip_2**: Boolean indicating if the track was only played briefly")
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

st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")
st.markdown("- ****: ")