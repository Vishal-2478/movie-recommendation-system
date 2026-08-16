import pandas as pd
import pytest

import recommender


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": [
                "Toy Story (1995)",
                "Toy Story 2 (1999)",
                "Batman (1989)",
                "Obscure Movie (2020)",
            ],
            "genres": ["Animation", "Animation", "Action", "Drama"],
        }
    )


@pytest.fixture
def ratings():
    """Three users rating movies 1-3; movie 4 is rated once so it is filtered out."""
    rows = []

    for user_id, base in ((1, 5.0), (2, 4.5), (3, 1.0)):
        rows.append({"userId": user_id, "movieId": 1, "rating": base})
        rows.append({"userId": user_id, "movieId": 2, "rating": base})
        rows.append({"userId": user_id, "movieId": 3, "rating": 6.0 - base})

    rows.append({"userId": 4, "movieId": 4, "rating": 3.0})

    return pd.DataFrame(rows)


@pytest.fixture
def low_thresholds(monkeypatch):
    """Lower the noise filters so the tiny fixtures survive `build_model`."""
    monkeypatch.setattr(recommender, "MIN_MOVIE_RATINGS", 1)
    monkeypatch.setattr(recommender, "MIN_USER_RATINGS", 1)


@pytest.fixture
def model(ratings, low_thresholds):
    """`(matrix, csr_data, knn, n_neighbors)` trained on the fixture ratings."""
    return recommender.build_model(ratings)
