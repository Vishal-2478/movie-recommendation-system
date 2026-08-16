"""Data loading and KNN recommendation logic for the movie recommender."""

import logging
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

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


def load_data():
    movies = _read_csv(MOVIES_FILE, {"movieId", "title"})
    ratings = _read_csv(RATINGS_FILE, {"movieId", "userId", "rating"})
    return movies, ratings


def active_index(ratings, group_column, min_count):
    """Index values of ``group_column`` with more than ``min_count`` ratings."""
    counts = ratings.groupby(group_column)["rating"].count()
    return counts[counts > min_count].index


def titles_for(movies, movie_id):
    """All titles registered for ``movie_id`` (empty when it is unknown)."""
    return movies.loc[movies["movieId"] == movie_id, "title"].values


def build_model(ratings):
    """Build the filtered movie-user matrix and fit the KNN model."""
    try:
        matrix = ratings.pivot(index="movieId", columns="userId", values="rating")
    except ValueError as exc:
        raise DataError(
            "ratings.csv contains duplicate (userId, movieId) pairs, "
            "so the rating matrix cannot be built."
        ) from exc

    matrix = matrix.fillna(0)
    matrix = matrix.loc[
        active_index(ratings, "movieId", MIN_MOVIE_RATINGS),
        active_index(ratings, "userId", MIN_USER_RATINGS),
    ]

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
    """Return the matched title and a DataFrame of similar movies.

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
        titles = titles_for(movies, recommended_movie_id)

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
