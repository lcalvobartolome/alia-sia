"""This class uses the funcionalities of the ALIA PIPELINE with the configurations already defined to be used directly in the API."""

import logging
from pathlib import Path

import alia_pipeline.nlp_pipeline
from alia_pipeline.nlp_pipeline.pipe import Pipe

_STOPS_DIR = Path(alia_pipeline.nlp_pipeline.__file__).parent / "stops"


class SIATools:
    def __init__(
        self,
        logger: logging.Logger = None,
    ) -> None:
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level="INFO")
            self._logger = logging.getLogger(__name__)

        self._pipe = Pipe(
            stw_files=sorted(_STOPS_DIR.glob("*.txt")),
            spaCy_model="es_core_news_lg",
            max_length=1_000_000,
            raw_text_cols=["text"],
            logger=self._logger,
        )

    def get_lemmas(self, text: str) -> list:
        return self._pipe.do_pipeline(text)

    def get_embedding(self, text: str) -> list:
        if not hasattr(self, "_embeddings_manager"):
            from alia_pipeline.nlp_pipeline.embeddings import EmbeddingsManager
            self._embeddings_manager = EmbeddingsManager(
                model_name="hiiamsid/sentence_similarity_spanish_es",
                max_seq_length=384,
                batch_size=32,
                logger=self._logger,
            )
        return self._embeddings_manager._model.encode([text])[0].tolist()


    