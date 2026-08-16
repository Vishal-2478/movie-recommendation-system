import logging

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

import recommender
from recommender import (
    DataError,
    MovieNotFoundError,
    MovieNotRatedEnoughError,
    active_index,
    build_model,
    get_recommendation,
    load_data,
    titles_for,
)


def write_csvs(tmp_path, monkeypatch, movies=None, ratings=None, raw=None):
    """Point the module at CSVs written into `tmp_path`."""
    movies_path = tmp_path / "movies.csv"
    ratings_path = tmp_path / "ratings.csv"

    if raw is not None:
        movies_path.write_text(raw.get("movies", ""))
        ratings_path.write_text(raw.get("ratings", ""))
    else:
        movies.to_csv(movies_path, index=False)
        ratings.to_csv(ratings_path, index=False)

    monkeypatch.setattr(recommender, "MOVIES_FILE", movies_path)
    monkeypatch.setattr(recommender, "RATINGS_FILE", ratings_path)
    return movies_path, ratings_path


class TestLoadData:
    def test_reads_both_csv_files(self, tmp_path, monkeypatch, movies, ratings):
        write_csvs(tmp_path, monkeypatch, movies, ratings)

        loaded_movies, loaded_ratings = load_data()

        pd.testing.assert_frame_equal(loaded_movies, movies)
        pd.testing.assert_frame_equal(loaded_ratings, ratings)

    def test_missing_file_raises_data_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(recommender, "MOVIES_FILE", tmp_path / "movies.csv")

        with pytest.raises(DataError, match="not found: movies.csv"):
            load_data()

    def test_empty_file_raises_data_error(self, tmp_path, monkeypatch):
        write_csvs(tmp_path, monkeypatch, raw={"movies": "", "ratings": ""})

        with pytest.raises(DataError, match="is empty: movies.csv"):
            load_data()

    def test_unreadable_file_raises_data_error(self, tmp_path, monkeypatch):
        unreadable = tmp_path / "movies.csv"
        unreadable.mkdir()
        monkeypatch.setattr(recommender, "MOVIES_FILE", unreadable)

        with pytest.raises(DataError, match="Could not read dataset file movies.csv"):
            load_data()

    def test_missing_column_raises_data_error(self, tmp_path, monkeypatch, ratings):
        write_csvs(
            tmp_path,
            monkeypatch,
            movies=pd.DataFrame({"movieId": [1], "name": ["Toy Story"]}),
            ratings=ratings,
        )

        with pytest.raises(DataError, match="missing required column\\(s\\): title"):
            load_data()

    def test_header_only_file_raises_data_error(self, tmp_path, monkeypatch):
        write_csvs(
            tmp_path,
            monkeypatch,
            raw={"movies": "movieId,title\n", "ratings": "userId,movieId,rating\n"},
        )

        with pytest.raises(DataError, match="movies.csv contains no rows"):
            load_data()


class TestActiveIndex:
    def test_keeps_keys_above_threshold(self, ratings):
        assert list(active_index(ratings, "movieId", 1)) == [1, 2, 3]
        assert list(active_index(ratings, "userId", 1)) == [1, 2, 3]

    def test_threshold_is_exclusive(self, ratings):
        """A key with exactly `min_count` ratings is dropped."""
        assert list(active_index(ratings, "movieId", 3)) == []

    def test_zero_threshold_keeps_everything(self, ratings):
        assert list(active_index(ratings, "movieId", 0)) == [1, 2, 3, 4]


class TestTitlesFor:
    def test_returns_title_for_known_id(self, movies):
        assert list(titles_for(movies, 3)) == ["Batman (1989)"]

    def test_returns_empty_for_unknown_id(self, movies):
        assert len(titles_for(movies, 999)) == 0


class TestBuildModel:
    def test_filters_noise_and_fits_model(self, model):
        matrix, csr_data, knn, _ = model

        assert list(matrix["movieId"]) == [1, 2, 3]
        assert list(matrix.index) == [0, 1, 2]
        assert issparse(csr_data)
        assert csr_data.shape == (3, 3)
        assert knn.metric == "cosine"

    def test_missing_ratings_become_zero(self, ratings, low_thresholds):
        matrix, csr_data, _, _ = build_model(ratings)

        assert matrix.isna().sum().sum() == 0
        np.testing.assert_allclose(
            csr_data.toarray(), matrix.drop(columns="movieId").values
        )

    def test_n_neighbors_capped_by_movie_count(self, model):
        _, _, knn, n_neighbors = model

        assert n_neighbors == 3
        assert knn.n_neighbors == 3

    def test_duplicate_user_movie_pair_raises_data_error(self, ratings, low_thresholds):
        duplicated = pd.concat([ratings, ratings.head(1)], ignore_index=True)

        with pytest.raises(DataError, match="duplicate"):
            build_model(duplicated)

    def test_fully_filtered_dataset_raises_data_error(self, ratings):
        with pytest.raises(DataError, match="No movies or users remain"):
            build_model(ratings)


class TestGetRecommendation:
    def recommend(self, movies, model, query):
        matrix, csr_data, knn, n_neighbors = model
        return get_recommendation(query, movies, matrix, csr_data, knn, n_neighbors)

    def test_returns_matched_title_and_similar_movies(self, movies, model):
        title, result = self.recommend(movies, model, "Toy Story")

        assert title == "Toy Story (1995)"
        assert list(result.columns) == ["Movie", "Similarity"]
        assert "Toy Story (1995)" not in list(result["Movie"])
        assert result.iloc[0]["Movie"] == "Toy Story 2 (1999)"

    def test_similarity_is_a_bounded_percentage(self, movies, model):
        _, result = self.recommend(movies, model, "Toy Story")

        for value in result["Similarity"]:
            assert value.endswith("%")
            assert 0.0 <= float(value.rstrip("%")) <= 100.0

    def test_identical_rating_pattern_scores_full_similarity(self, movies, model):
        _, result = self.recommend(movies, model, "Toy Story")

        assert result.iloc[0]["Similarity"] == "100.00%"

    def test_row_count_is_capped_by_available_movies(self, movies, model):
        _, result = self.recommend(movies, model, "Toy Story")

        assert len(result) == 2

    @pytest.mark.parametrize("query", ["toy story", "TOY STORY"])
    def test_search_is_case_insensitive(self, movies, model, query):
        title, _ = self.recommend(movies, model, query)

        assert title == "Toy Story (1995)"

    def test_query_is_matched_literally_not_as_regex(self, movies, model):
        title, _ = self.recommend(movies, model, "Batman (1989)")

        assert title == "Batman (1989)"

        with pytest.raises(MovieNotFoundError):
            self.recommend(movies, model, "Batman.1989")

    def test_unknown_title_raises(self, movies, model):
        with pytest.raises(MovieNotFoundError, match="Nonexistent Film"):
            self.recommend(movies, model, "Nonexistent Film")

    def test_noise_filtered_title_raises(self, movies, model):
        """`Obscure Movie` is in movies.csv but was dropped from the matrix."""
        with pytest.raises(MovieNotRatedEnoughError) as excinfo:
            self.recommend(movies, model, "Obscure Movie")

        assert excinfo.value.title == "Obscure Movie (2020)"

    def test_neighbour_without_a_title_is_skipped(self, movies, model, caplog):
        untitled = movies[movies["movieId"] != 2]

        with caplog.at_level(logging.WARNING):
            _, result = self.recommend(untitled, model, "Toy Story")

        assert list(result["Movie"]) == ["Batman (1989)"]
        assert "has ratings but no title" in caplog.text

    def test_raises_when_no_neighbour_has_a_title(self, movies, model):
        only_query_movie = movies[movies["movieId"] == 1]

        with pytest.raises(MovieNotRatedEnoughError):
            self.recommend(only_query_movie, model, "Toy Story")
