import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"

MIN_MOVIE_RATINGS = 10
MIN_USER_RATINGS = 50
N_RECOMMENDATIONS = 10


class DataError(Exception):
    """Raised when the dataset is missing, unreadable or malformed."""


class MovieNotFoundError(Exception):
    """Raised when no movie title matches the search term."""


class MovieNotRatedEnoughError(Exception):
    """Raised when a matched movie was filtered out for having too few ratings."""

    def __init__(self, title):
        super().__init__(
            f"'{title}' does not have enough ratings "
            f"(at least {MIN_MOVIE_RATINGS} required) to compute recommendations."
        )
        self.title = title


st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Recommend similar movies using KNN Collaborative Filtering")


def _read_csv(path, required_columns):
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataError(f"Dataset file not found: {path.name}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"Dataset file is empty: {path.name}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataError(f"Could not read dataset file {path.name}: {exc}") from exc

    missing = required_columns.difference(frame.columns)
    if missing:
        raise DataError(
            f"{path.name} is missing required column(s): {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise DataError(f"{path.name} contains no rows.")
    return frame


@st.cache_data
def load_data():
    movies = _read_csv(MOVIES_FILE, {"movieId", "title"})
    ratings = _read_csv(RATINGS_FILE, {"movieId", "userId", "rating"})
    return movies, ratings


@st.cache_resource
def build_model(_ratings):
    """Build the filtered movie-user matrix and fit the KNN model."""
    try:
        matrix = _ratings.pivot(index="movieId", columns="userId", values="rating")
    except ValueError as exc:
        raise DataError(
            "ratings.csv contains duplicate (userId, movieId) pairs, "
            "so the rating matrix cannot be built."
        ) from exc

    matrix = matrix.fillna(0)

    ratings_per_movie = _ratings.groupby("movieId")["rating"].count()
    ratings_per_user = _ratings.groupby("userId")["rating"].count()

    matrix = matrix.loc[ratings_per_movie[ratings_per_movie > MIN_MOVIE_RATINGS].index]
    matrix = matrix.loc[:, ratings_per_user[ratings_per_user > MIN_USER_RATINGS].index]

    if matrix.empty:
        raise DataError(
            "No movies or users remain after filtering out sparsely rated entries. "
            "The dataset is too small for collaborative filtering."
        )

    csr_data = csr_matrix(matrix.values)
    matrix = matrix.reset_index()

    n_neighbors = min(N_RECOMMENDATIONS + 1, csr_data.shape[0])
    knn = NearestNeighbors(
        metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1
    )
    knn.fit(csr_data)

    logger.info(
        "Fitted KNN model on %d movies x %d users", matrix.shape[0], matrix.shape[1] - 1
    )
    return matrix, csr_data, knn, n_neighbors


def get_recommendation(movie_name, movies, matrix, csr_data, knn, n_neighbors):
    """Return a DataFrame of similar movies.

    Raises:
        MovieNotFoundError: no title matches ``movie_name``.
        MovieNotRatedEnoughError: the matched movie was filtered out of the matrix.
    """
    # regex=False so that titles containing regex metacharacters are matched literally
    matches = movies[
        movies["title"].str.contains(movie_name, case=False, na=False, regex=False)
    ]

    if matches.empty:
        raise MovieNotFoundError(f"No movie title matches '{movie_name}'.")

    movie_id = matches.iloc[0]["movieId"]
    title = matches.iloc[0]["title"]

    positions = matrix.index[matrix["movieId"] == movie_id]
    if positions.empty:
        raise MovieNotRatedEnoughError(title)

    distances, indices = knn.kneighbors(
        csr_data[positions[0]], n_neighbors=n_neighbors
    )

    recommendations = []
    for idx, distance in zip(indices.flatten()[1:], distances.flatten()[1:]):
        recommended_movie_id = matrix.iloc[idx]["movieId"]
        titles = movies.loc[movies["movieId"] == recommended_movie_id, "title"].values

        if len(titles) == 0:
            # ratings.csv references a movieId absent from movies.csv
            logger.warning(
                "movieId %s has ratings but no title in %s; skipping",
                recommended_movie_id,
                MOVIES_FILE.name,
            )
            continue

        recommendations.append(
            {"Movie": titles[0], "Similarity": f"{max(0.0, 1 - distance) * 100:.2f}%"}
        )

    if not recommendations:
        raise MovieNotRatedEnoughError(title)

    return title, pd.DataFrame(recommendations)


try:
    movies, ratings = load_data()
    final_dataset, csr_data, knn, n_neighbors = build_model(ratings)
except DataError as exc:
    logger.exception("Failed to initialise the recommendation model")
    st.error(f"Could not start the recommender: {exc}")
    st.stop()

st.markdown("---")

movie_name = st.text_input(
    "Enter Movie Name", placeholder="Example: Batman, Toy Story, Avatar"
)

if st.button("🎬 Recommend"):

    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")

    else:
        try:
            title, result = get_recommendation(
                movie_name.strip(), movies, final_dataset, csr_data, knn, n_neighbors
            )
        except MovieNotFoundError as exc:
            st.error(str(exc))
        except MovieNotRatedEnoughError as exc:
            st.warning(str(exc))
        except Exception:
            # Log the full traceback, then let Streamlit surface the failure
            # instead of silently rendering an empty result.
            logger.exception("Unexpected failure while recommending for %r", movie_name)
            st.error("Something went wrong while generating recommendations.")
            raise
        else:
            st.success(f"Top {len(result)} movies similar to '{title}'")
            st.dataframe(result, use_container_width=True, hide_index=True)
