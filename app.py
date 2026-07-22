import streamlit as st
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Recommend similar movies using KNN Collaborative Filtering")


@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings


movies, ratings = load_data()

final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating")

final_dataset.fillna(0, inplace=True)

# -----------------------------
# Remove Noisy Movies & Users
# -----------------------------

# Number of ratings received by each movie
no_user_voted = ratings.groupby("movieId")["rating"].count()

# Number of ratings given by each user
no_movies_voted = ratings.groupby("userId")["rating"].count()

# Keep movies with at least 10 ratings
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

# Keep users who rated at least 50 movies
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]


# -----------------------------
# Convert to Sparse Matrix
# -----------------------------

csr_data = csr_matrix(final_dataset.values)

final_dataset.reset_index(inplace=True)


# -----------------------------
# Train KNN Model
# -----------------------------

knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11, n_jobs=-1)

knn.fit(csr_data)

# -----------------------------
# Recommendation Function
# -----------------------------


def get_recommendation(movie_name):

    # Search movie (case-insensitive)
    movie_list = movies[movies["title"].str.contains(movie_name, case=False, na=False)]

    if movie_list.empty:
        return None

    movie_id = movie_list.iloc[0]["movieId"]

    # Check if movie exists after filtering
    if movie_id not in final_dataset["movieId"].values:
        return None

    movie_index = final_dataset[final_dataset["movieId"] == movie_id].index[0]

    distances, indices = knn.kneighbors(csr_data[movie_index], n_neighbors=11)

    recommendations = []

    for idx, distance in zip(indices.flatten()[1:], distances.flatten()[1:]):

        recommended_movie_id = final_dataset.iloc[idx]["movieId"]

        title = movies.loc[movies["movieId"] == recommended_movie_id, "title"].values[0]

        recommendations.append(
            {"Movie": title, "Similarity": f"{(1-distance)*100:.2f}%"}
        )

    return pd.DataFrame(recommendations)


st.markdown("---")

movie_name = st.text_input(
    "Enter Movie Name", placeholder="Example: Batman, Toy Story, Avatar"
)

if st.button("🎬 Recommend"):

    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")

    else:

        result = get_recommendation(movie_name)

        if result is None:
            st.error("Movie not found in dataset.")
        else:
            st.success("Top 10 Similar Movies")
            st.dataframe(result, use_container_width=True, hide_index=True)
