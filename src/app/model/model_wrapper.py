import enum
import os
import pickle
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
    KERAS_VECTORIZER = "keras-vectorizer"


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

    vectorizer: Optional[Any] = None  # instance of keras sequencer or TextVectorization
    vectorizer_technique: Optional[VectorizeTechnique] = None
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

    @field_validator("vectorizer_technique", mode="before")
    def vectorize_technique_validator(cls, value: Optional[str]):
        match value:
            case VectorizeTechnique.N_GRAM.value:
                return VectorizeTechnique.N_GRAM

            case VectorizeTechnique.SEQUENCE.value:
                return VectorizeTechnique.SEQUENCE

            case VectorizeTechnique.KERAS_VECTORIZER.value:
                return VectorizeTechnique.KERAS_VECTORIZER

            case _:
                raise AIXInvalidVectorTypeError("not recognized vectorization technique: '{}'".format(value))

    @field_validator("vectorization_hyperparameters", mode="before")
    def vectorization_hyperparameters_validator(cls, value: Optional[str]):
        return cls.parse_parameters(value)

    @field_validator("model_hyperparameters", mode="before")
    def model_hyperparameters_validator(cls, value: Optional[str]):
        return cls.parse_parameters(value)

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

        if self.vectorizer_technique == VectorizeTechnique.N_GRAM:
            transformed = self.vectorizer.transform(x)
            return np.array(transformed.toarray())

        elif self.vectorizer_technique == VectorizeTechnique.KERAS_VECTORIZER:
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

        match self.vectorizer_technique:

            case VectorizeTechnique.N_GRAM:
                transformed = self.vectorizer.transform(x)
                transformed = np.array(transformed.toarray())

            case VectorizeTechnique.SEQUENCE:
                transformed = self.vectorizer(x)

            case VectorizeTechnique.KERAS_VECTORIZER:
                transformed = self.vectorizer(x)

            case _:
                transformed = x

        return self.model.predict(transformed).flatten()

    def build_model(self):
        self.model_builded = False

        if not os.path.exists(self.model_path):
            raise AIXModelNotFoundError("model file do not exist: '{}'".format(self.model_path))

        self.model = keras.saving.load_model(self.model_path)
        try:
            self.vectorizer = self.load(self.vectorizer_path)
        except Exception:
            raise AIXVectorizerNotFoundError("vectorizer file do not exist: '{}'".format(self.model_path))

        self._load_sequences()
        self.model_builded = True

    def _load_sequences(self):
        # Determine file type and load sequences
        if self.sequences_path.endswith(".npy"):
            sequences = np.load(self.sequences_path)
        elif self.sequences_path.endswith(".pkl"):
            with open(self.sequences_path, "rb") as f:
                sequences = pickle.load(f)
        else:
            raise AIXSequencesError(
                f"Unsupported file format: {self.sequences_path}. " "Only .npy and .pkl files are supported"
            )

        match self.vectorizer_technique:
            case VectorizeTechnique.N_GRAM:
                self.sequences = sequences

                if type(self.sequences).__name__ == "csr_matrix":
                    raise AIXSequencesError(
                        "not valid sequences, they should be a list of strings, not vectors '{}'".format(
                            self.sequences_path
                        )
                    )

            case VectorizeTechnique.SEQUENCE:
                self.sequences = sequences

                if self.reverse_sequences:
                    self.sequences = self._reverse_sequences(self.sequences)

            case VectorizeTechnique.KERAS_VECTORIZER:
                self.sequences = np.array(sequences)

                config, weights = self.vectorizer["config"], self.vectorizer["weights"]
                self.vectorizer = keras.layers.TextVectorization.from_config(config)
                self.vectorizer.set_weights(weights)

                if self.reverse_sequences:
                    self.sequences = self._reverse_sequences(self.sequences)

    def _reverse_sequences(self, sequences):
        if not hasattr(self.vectorizer, "get_vocabulary"):
            return sequences

        vocab = self.vectorizer.get_vocabulary()
        reversed_sequences = []

        for seq in sequences:
            words = [vocab[idx] for idx in seq if idx > 0 and idx < len(vocab)]
            reversed_sequences.append(" ".join(words))

        return np.array(reversed_sequences)
