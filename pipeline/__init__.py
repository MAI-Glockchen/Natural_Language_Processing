# -----------------------------
# Pipeline package initializer
# Expose modules that exist in the current codebase
# -----------------------------

from .mock_articles import MOCK_ARTICLES
from .topic_inference import infer_topic
from .topic_vector_pipeline import TopicVectorPipeline
from .vector_index import PassageVectorIndex

__all__ = [
    "MOCK_ARTICLES",
    "infer_topic",
    "TopicVectorPipeline",
    "PassageVectorIndex",
]
