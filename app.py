import logging

import streamlit as st

from recommender import (
    MAX_QUERY_LENGTH,
    DataError,
    MovieNotFoundError,
    MovieNotRatedEnoughError,
    build_model,
    get_recommendation,
    load_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Recommend similar movies using KNN Collaborative Filtering")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(_ratings):
    return build_model(_ratings)


try:
    movies, ratings = get_data()
    final_dataset, csr_data, knn, n_neighbors = get_model(ratings)
except DataError as exc:
    logger.exception("Failed to initialise the recommendation model")
    st.error(f"Could not start the recommender: {exc}")
    st.stop()

st.markdown("---")

movie_name = st.text_input(
    "Enter Movie Name",
    placeholder="Example: Batman, Toy Story, Avatar",
    max_chars=MAX_QUERY_LENGTH,
)

if st.button("🎬 Recommend"):

    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")

    else:
        try:
            title, result = get_recommendation(
                movie_name, movies, final_dataset, csr_data, knn, n_neighbors
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
