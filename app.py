import streamlit as st
import pandas as pd
""
from main import load_data, load_user_history, recommend_songs

st.set_page_config(page_title="Музичні рекомендації", layout="wide")

st.title("🎵 Рекомендаційна система музичного контенту")


data = load_data("data.csv")
users = pd.read_csv("users.csv")
user_history = load_user_history("user_favorites.csv")

with st.expander("📌 Інструкція"):
    st.write(
        "Виберіть профіль користувача або залиште 'Пустий користувач', щоб ввести свої треки вручну. "
        "Система знайде треки зі схожими характеристиками та популярністю."
    )

st.subheader("Виберіть профіль або введіть свої треки")
user_options = ["Empty user"] + users["name"].tolist()
selected_user = st.selectbox("Користувач", user_options, index=0)

favorite_defaults = []
favorites = pd.DataFrame()
if selected_user != "Empty user":
    selected_id = users.loc[users["name"] == selected_user, "user_id"].iloc[0]
    favorites = user_history[user_history["user_id"] == selected_id]
    favorite_defaults = favorites.head(3).to_dict(orient="records")

def default_field(index: int, field: str, fallback=""):
    if len(favorite_defaults) > index:
        return favorite_defaults[index].get(field, fallback)
    return fallback

col1, col2 = st.columns(2)
with col1:
    song1 = st.text_input("Пісня 1", default_field(0, "name"))
    artist1 = st.text_input("Артист 1", default_field(0, "artist"))
with col2:
    song2 = st.text_input("Пісня 2", default_field(1, "name"))
    artist2 = st.text_input("Артист 2", default_field(1, "artist"))

song3 = st.text_input("Пісня 3 (необов'язково)", default_field(2, "name"))
artist3 = st.text_input("Артист 3", default_field(2, "artist"))

if selected_user != "Empty user":
    with st.expander("Улюблені треки профілю", expanded=True):
        st.table(
            favorites[["name", "artist", "year", "rating"]]
            .rename(columns={"name": "Пісня", "artist": "Артист", "year": "Рік", "rating": "Оцінка"})
        )

strategy = st.selectbox(
    "Стратегія рекомендацій",
    [
        "User-based CF",
        "Гібридна (контент + популярність)",
        "Контент тільки",
        "Популярні треки",
    ],
)

if st.button("Отримати рекомендації"):
    year1 = default_field(0, "year", 2015)
    year2 = default_field(1, "year", 2016)
    year3 = default_field(2, "year", 2018)

    input_songs = []
    if song1.strip():
        input_songs.append({"name": song1, "artist": artist1, "year": int(year1)})
    if song2.strip():
        input_songs.append({"name": song2, "artist": artist2, "year": int(year2)})
    if song3.strip():
        input_songs.append({"name": song3, "artist": artist3, "year": int(year3)})

    recs = []
    if selected_user != "Empty user" and strategy == "User-based CF":
        selected_id = users.loc[users["name"] == selected_user, "user_id"].iloc[0]
        with st.spinner("Генеруємо рекомендації для користувача..."):
            recs = recommend_songs(
                [],
                strategy,
                data,
                n_songs=10,
                user_id=int(selected_id),
                user_history=user_history,
            )
    else:
        if not input_songs and selected_user != "Empty user":
            selected_id = users.loc[users["name"] == selected_user, "user_id"].iloc[0]
            profile_songs = user_history[user_history["user_id"] == selected_id][["name", "artist", "year"]].to_dict(orient="records")
            input_songs = profile_songs
        if not input_songs:
            st.warning("Будь ласка, введіть хоча б одну пісню для рекомендацій.")
            recs = []
        else:
            with st.spinner("Генеруємо рекомендації..."):
                recs = recommend_songs(input_songs, strategy, data, n_songs=10)

    if recs:
        rec_df = pd.DataFrame(recs)
        st.success(f"Знайдено {len(rec_df)} рекомендацій!")
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.error("Не вдалось знайти рекомендації за введеними треками чи профілем.")

st.markdown("---")

with st.expander("Інформація про набір даних"):
    st.write(f"Кількість треків: {len(data)}")
    top_artists = data["artists"].value_counts().head(10).reset_index()
    top_artists.columns = ["Артист", "Кількість треків"]
    st.dataframe(top_artists, use_container_width=True)

with st.expander("Приклад даних"):
    st.dataframe(data[["name", "artists", "year", "popularity"]].head(10), use_container_width=True)
