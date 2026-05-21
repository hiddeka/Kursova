## Запуск проекту

Для запуску прототипу користувачеві потрібно встановити Python та залежності:

- pandas
- numpy
- scikit-learn
- streamlit
- python-dotenv
- spotipy

Після цього файли:

- `main.py`
- `app.py`
- `users.csv`
- `user_favorites.csv`
- `data.csv`

мають бути розміщені в одному каталозі.

Використання Spotify API

Для використовуватання Spotify API, потрібно створити файл .env зі змінними:
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret.

Streamlit-застосунок запускається командою:

```bash
streamlit run app.py

