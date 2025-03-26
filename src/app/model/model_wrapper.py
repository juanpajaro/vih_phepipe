import enum
import os
from typing import Any, Optional

import numpy as np
import sklearn
from pydantic import field_validator
from tensorflow import keras

from app.model.abstract_wrapper import AbstractWrapper
from app.model.errors import (
    AIXInvalidVectorTypeError,
    AIXModelNotFoundError,
    AIXSequencesError,
    AIXVectorizerNotFoundError,
)


class ModelType(enum.Enum):
    KERNEL = "kernel"
    MPL = "mpl"
    TRANSFORMER = "transformer"


class VectorizeTechnique(enum.Enum):
    N_GRAM = "n-gram"
    SEQUENCE = "sequence"
    OTHER_SEQUENCE = "other-sequence"


class ModelWrapper(AbstractWrapper):
    name: Optional[str]
    model: Optional[Any] = None  # instance of keras model
    model_type: Optional[ModelType] = None
    folder: Optional[str]
    model_path: Optional[str]
    model_hyperparameters: Optional[Any]

    model_builded: Optional[bool] = False
    has_preloaded_sequences: Optional[bool] = False
    reverse_sequences: bool = False

    vectorizer: Optional[Any] = None  # instance of keras sequencer
    vectorizer_type: Optional[VectorizeTechnique] = None
    vectorizer_path: Optional[str]
    vectorization_hyperparameters: Optional[Any]

    sequences: Optional[Any] = None
    sequences_path: Optional[str]

    @field_validator("model_path", mode="before")
    def model_path_validator(cls, value: Optional[str]):
        return cls.path_validator(value, "model")

    @field_validator("vectorizer_path", mode="before")
    def vectorizer_path_validator(cls, value: Optional[str]):
        return cls.path_validator(value, "vectorizer")

    @field_validator("sequences_path", mode="before")
    def sequences_path_validator(cls, value: Optional[str]):
        return cls.path_validator(value, "sequences")

    @field_validator("vectorization_hyperparameters", mode="before")
    def vectorization_hyperparameters_validator(cls, value: Optional[str]):
        allowed_values = [
            VectorizeTechnique.N_GRAM.value,
            VectorizeTechnique.SEQUENCE.value,
            VectorizeTechnique.OTHER_SEQUENCE.value,
        ]
        if not value["vectorize_technique"]:
            return value

        if value["vectorize_technique"] not in allowed_values:
            raise AIXInvalidVectorTypeError(
                "not recognized vectorization technique: '{}'".format(value["vectorize_technique"])
            )

        return value

    def transform(self, x):
        """
        The function `transform` checks if a vectorizer is instantiated and applies different
        transformation techniques based on the vector type.

        :param x: It looks like the code you provided is a method named `transform` within a class, and
        it takes a parameter `x`. The method checks if a vectorizer is instantiated for the class
        instance (`self`) and then transforms the input `x` based on the vectorization technique
        specified by `self
        :return: The `transform` method returns the transformed input `x` based on the vectorization
        technique specified in `self.vector_type`. If the vector type is `N_GRAM`, it transforms the
        input using the vectorizer and returns it as a NumPy array. If the vector type is `SEQUENCE`, it
        directly returns the result of the vectorizer applied to the input. If the vector type is
        """
        if self.vectorizer == None:
            raise AIXVectorizerNotFoundError(f"vectorizer not instantiated for {self.name}")

        if self.vectorizer_type == VectorizeTechnique.N_GRAM:
            transformed = self.vectorizer.transform(x)
            return np.array(transformed.toarray())

        elif self.vectorizer_type == VectorizeTechnique.SEQUENCE:
            return self.vectorizer(x)

        return x

    def predict(self, x):
        """
        This Python function predicts outcomes using a specified vectorization technique and a machine
        learning model.

        :param x: It looks like the `predict` method in the code snippet is used to make predictions
        using a machine learning model. The input parameter `x` is the data that you want to make
        predictions on
        :return: The predict method returns the predictions made by the model on the input data x after
        transforming it using the appropriate vectorization technique specified in the code. The
        predictions are flattened before being returned.
        """
        transformed = []
        if self.model == None:
            return None

        match self.vectorizer_type:

            case VectorizeTechnique.N_GRAM:
                transformed = self.vectorizer.transform(x)
                transformed = np.array(transformed.toarray())

            case VectorizeTechnique.SEQUENCE:
                transformed = self.vectorizer(x)

            case VectorizeTechnique.OTHER_SEQUENCE:
                transformed = keras.utils.pad_sequences(
                    self.vectorizer.texts_to_sequences(x),
                    maxlen=self.vectorization_hyperparameters["MAX_SEQUENCE_LENGTH"],
                    dtype="int32",
                    padding="pre",
                    truncating="pre",
                    value=0,
                )

            case _:
                transformed = x

        return self.model.predict(transformed).flatten()

    def build_new_vectorizer(self, x):
        if self.vectorizer is not None:
            return

        match self.vectorization_hyperparameters["vectorize_technique"]:

            case VectorizeTechnique.N_GRAM.value:
                self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                    analyzer=self.vectorization_hyperparameters["TOKEN_MODE"],
                    min_df=self.vectorization_hyperparameters["MIN_DOCUMENT_FREQUENCY"],
                    ngram_range=self.vectorization_hyperparameters["NGRAM_RANGE"],
                    max_features=self.vectorization_hyperparameters["TOP_K"],
                )
                self.vectorizer_type = VectorizeTechnique.N_GRAM
                self.vectorizer.fit_transform(x)

            case VectorizeTechnique.SEQUENCE.value:
                self.vectorizer = keras.layers.TextVectorization(
                    standardize="lower_and_strip_punctuation",
                    pad_to_max_tokens=True,
                    max_tokens=self.vectorization_hyperparameters["TOP_K"],
                    output_sequence_length=self.vectorization_hyperparameters["MAX_SEQUENCE_LENGTH"],
                )
                self.vectorizer_type = VectorizeTechnique.SEQUENCE
                self.vectorizer.adapt(x)

            case _:
                self.vectorizer_type = VectorizeTechnique.OTHER_SEQUENCE

    def build_model(self):
        self.model_builded = False
        if not os.path.exists(self.model_path):
            raise AIXModelNotFoundError("model file do not exist: '{}'".format(self.model_path))

        self.model = keras.saving.load_model(self.model_path)
        try:
            self.vectorizer = self.load(self.vectorizer_path)
        except Exception:
            raise AIXVectorizerNotFoundError("vectorizer file do not exist: '{}'".format(self.model_path))

        match self.vectorization_hyperparameters["vectorize_technique"]:

            case VectorizeTechnique.N_GRAM.value:
                self.vectorizer_type = VectorizeTechnique.N_GRAM
                self.sequences = self.load(self.sequences_path)

                if type(self.sequences).__name__ == "csr_matrix":
                    raise AIXSequencesError(
                        "not valid sequences, they should be a list of strings, not vectors '{}'".format(
                            self.sequences_path
                        )
                    )

            case VectorizeTechnique.SEQUENCE.value:
                self.vectorizer_type = VectorizeTechnique.SEQUENCE
                self.sequences = self.load(self.sequences_path)

            case VectorizeTechnique.OTHER_SEQUENCE.value:
                self.vectorizer_type = VectorizeTechnique.OTHER_SEQUENCE
                self.sequences = self.load(self.sequences_path)
                self.sequences = np.array(self.sequences)

            case _:
                raise AIXInvalidVectorTypeError(
                    "not recognized vectorization technique: '{}'".format(
                        self.vectorization_hyperparameters["vectorize_technique"]
                    )
                )

        self.model_builded = True
