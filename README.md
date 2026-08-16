# 🎬 Movie Recommendation System

A Movie Recommendation System built using **Python, Streamlit, Scikit-learn, Pandas, NumPy, and SciPy** that recommends similar movies based on **Item-Based Collaborative Filtering** using the **K-Nearest Neighbors (KNN)** algorithm and **Cosine Similarity**.

The application provides movie recommendations by analyzing historical user ratings from the MovieLens dataset.

---

## 🚀 Live Demo

https://knn-movie-recommendation-system.streamlit.app/

---

## 📸 Screenshot

<img width="608" height="244" alt="image" src="https://github.com/user-attachments/assets/95873578-a87b-426d-aa5b-6c6b56f79086" />


---

## ✨ Features

- 🎬 Search movies by title
- 🤖 Item-Based Collaborative Filtering
- 📊 K-Nearest Neighbors (KNN) recommendation engine
- 📐 Cosine Similarity for finding similar movies
- ⚡ Fast recommendations using Sparse Matrix representation
- 🔍 Case-insensitive movie search
- 🌐 Interactive Streamlit web interface
- 📈 Top 10 similar movie recommendations

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Scikit-learn

---

## 📂 Dataset

This project uses the **MovieLens Latest Small Dataset**.

Dataset contains:

- 100,000+ Ratings
- 9,700+ Movies
- 600+ Users

Source:

https://grouplens.org/datasets/movielens/

---

## 🧠 Recommendation Algorithm

This project uses **Item-Based Collaborative Filtering**.

### Workflow

1. Load MovieLens dataset
2. Create a Movie-User Rating Matrix
3. Replace missing ratings with 0
4. Remove noisy movies and inactive users
5. Convert the matrix into a Sparse Matrix (CSR)
6. Train a KNN model using Cosine Similarity
7. Retrieve Top 10 most similar movies

---

## 📊 Project Structure

```
Movie-Recommendation-System/
│
├── app.py                  # Streamlit UI
├── recommender.py          # Data loading + KNN recommendation logic
├── tests/
│   ├── conftest.py
│   └── test_recommender.py
├── movies.csv
├── ratings.csv
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Vishal-2478/movie-recommendation-system.git
```

Go to the project directory

```bash
cd Movie-Recommendation-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🧪 Tests

Install the development dependencies and run the suite with coverage:

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage for `recommender.py` is reported automatically.

---

## 📌 Future Improvements

- Movie posters using TMDB API
- Genre-based filtering
- Hybrid Recommendation System
- Personalized recommendations
- User authentication
- Recommendation history
- Deep Learning based recommender

---

## 📈 Sample Output

Input

```
Toy Story
```

Output

```
Toy Story 2
Monsters Inc.
Finding Nemo
Shrek
Bug's Life
...
```

---

## 💡 Learning Outcomes

- Collaborative Filtering
- Recommendation Systems
- K-Nearest Neighbors (KNN)
- Cosine Similarity
- Sparse Matrix Optimization
- Data Preprocessing
- Streamlit Deployment

---


---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
