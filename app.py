import pandas as pd
import streamlit as st

from main import (
    authenticate_user,
    get_user_ratings,
    load_data,
    load_user_history,
    recommend_songs,
    register_user,
    save_user_rating,
    search_tracks,
)


st.set_page_config(page_title="Музичні рекомендації", layout="wide")

data = load_data("data.csv")

if "user" not in st.session_state:
    st.session_state.user = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []


def logout():
    st.session_state.user = None
    st.session_state.search_results = []


st.title("Музична рекомендаційна система")

if st.session_state.user is None:
    login_tab, register_tab = st.tabs(["Вхід", "Реєстрація"])

    with login_tab:
        with st.form("login_form"):
            login_name = st.text_input("Ім'я користувача")
            login_password = st.text_input("Пароль", type="password")
            login_submitted = st.form_submit_button("Увійти")

        if login_submitted:
            user = authenticate_user(login_name, login_password)
            if user is None:
                st.error("Неправильне ім'я користувача або пароль.")
            else:
                st.session_state.user = user
                st.rerun()

    with register_tab:
        with st.form("register_form"):
            register_name = st.text_input("Нове ім'я користувача")
            register_password = st.text_input("Новий пароль", type="password")
            register_submitted = st.form_submit_button("Зареєструватися")

        if register_submitted:
            result = register_user(register_name, register_password)
            if result["ok"]:
                st.success(result["message"])
                st.session_state.user = {"user_id": result["user_id"], "name": result["name"]}
                st.rerun()
            else:
                st.error(result["message"])

    st.stop()


user = st.session_state.user
top_left, top_right = st.columns([3, 1])
with top_left:
    st.subheader(f"Користувач: {user['name']}")
with top_right:
    st.button("Вийти", on_click=logout, use_container_width=True)

search_tab, recommendations_tab = st.tabs(["Пошук і оцінки", "Рекомендації"])

with search_tab:
    st.subheader("Знайти трек")
    with st.form("search_form"):
        query = st.text_input("Назва треку або артист")
        search_submitted = st.form_submit_button("Шукати")

    if search_submitted:
        with st.spinner("Шукаємо треки..."):
            st.session_state.search_results = search_tracks(query, data, limit=10)

    if st.session_state.search_results:
        st.write("Результати пошуку")
        for index, track in enumerate(st.session_state.search_results):
            cols = st.columns([4, 1, 1, 1])
            title = f"{track['name']} - {track['artist']}"
            cols[0].markdown(f"**{title}**")
            cols[0].caption(f"{track['year']} · популярність {track['popularity']} · {track['source']}")
            rating = cols[1].selectbox(
                "Оцінка",
                [5, 4, 3, 2, 1],
                key=f"rating_{index}",
                label_visibility="collapsed",
            )
            if cols[2].button("Оцінити", key=f"save_rating_{index}", use_container_width=True):
                result = save_user_rating(user["user_id"], track, rating)
                if result["ok"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
            cols[3].write("")
    else:
        st.info("Введіть назву пісні або артиста, щоб знайти треки для оцінювання.")

    ratings = get_user_ratings(user["user_id"])
    st.subheader("Ваші оцінені треки")
    if ratings.empty:
        st.warning("Поки немає оцінок. Оцініть хоча б один трек перед рекомендаціями.")
    else:
        st.dataframe(
            ratings[["name", "artist", "year", "rating"]].sort_values("rating", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

with recommendations_tab:
    ratings = get_user_ratings(user["user_id"])
    st.subheader("Рекомендації")

    strategy = st.selectbox(
        "Стратегія рекомендацій",
        [
            "Гібридна (контент + колаборативна)",
            "Контент тільки",
            "Популярні треки",
            "User-based CF",
        ],
    )
    max_recommendations = st.number_input(
        "Максимальна кількість рекомендацій",
        min_value=1,
        max_value=30,
        value=30,
        step=1,
    )

    if ratings.empty:
        st.warning("Спочатку знайдіть треки у пошуку та поставте їм оцінки.")
    elif st.button("Отримати рекомендації", use_container_width=True):
        input_songs = ratings[["name", "artist", "year", "rating"]].to_dict(orient="records")
        user_history = load_user_history("user_favorites.csv")
        with st.spinner("Генеруємо рекомендації..."):
            recs = recommend_songs(
                input_songs,
                strategy,
                data,
                n_songs=int(max_recommendations),
                user_id=int(user["user_id"]),
                user_history=user_history,
            )

        if recs:
            rec_df = pd.DataFrame(recs)
            st.success(f"Знайдено {len(rec_df)} рекомендацій.")
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.error("Не вдалося знайти рекомендації за вашими оцінками.")
