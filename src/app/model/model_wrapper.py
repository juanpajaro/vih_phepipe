import enum
import os
import pickle
from typing import Any, Optional

import numpy as np
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
    """
    A wrapper class for machine learning models, particularly for handling text data processing and prediction.

    This class encapsulates the functionality needed to load, prepare, and use machine learning models
    with different vectorization techniques. It supports various types of models and vectorization methods,
    and handles sequence data processing for text-based machine learning tasks.

    Attributes:
        name: Optional name identifier for the model
        model: The actual Keras model instance
        model_type: Type of the model (KERNEL, MPL, TRANSFORMER)
        folder: Directory where model files are stored
        model_path: Path to the model file
        model_hyperparameters: Configuration parameters for the model
        model_builded: Flag indicating if the model has been successfully built
        has_preloaded_sequences: Flag indicating if sequences have been preloaded
        reverse_sequences: Flag to determine if sequences should be reversed
        vectorizer: Instance of text vectorizer (e.g., Keras sequencer or TextVectorization)
        vectorizer_technique: Technique used for vectorization (N_GRAM, SEQUENCE, KERAS_VECTORIZER)
        vectorizer_path: Path to the vectorizer file
        vectorization_hyperparameters: Configuration parameters for vectorization
        sequences: The sequence data used for model training/testing
        sequences_path: Path to the sequences file
    """

    name: Optional[str]
    model: Optional[Any] = None  # instance of keras model
    model_type: Optional[ModelType] = None
    folder: Optional[str]
    model_path: Optional[str]
    model_hyperparameters: Optional[Any]

    model_builded: Optional[bool] = False
    has_preloaded_sequences: Optional[bool] = False
    reverse_sequences: Optional[bool] = False
    clean_empty_sequences: Optional[bool] = True

    vectorizer: Optional[Any] = None  # instance of keras sequencer or TextVectorization
    vectorizer_technique: Optional[VectorizeTechnique] = None
    vectorizer_path: Optional[str]
    vectorization_hyperparameters: Optional[Any]

    sequences: Optional[Any] = None
    sequences_path: Optional[str]

    @field_validator("model_path", mode="before")
    def model_path_validator(cls, value: Optional[str]):
        """
        Validates the model path to ensure it exists and is properly formatted.

        Args:
            value: The path to validate

        Returns:
            The validated path

        Raises:
            Validation error if the path is invalid
        """
        return cls.path_validator(value, "model")

    @field_validator("vectorizer_path", mode="before")
    def vectorizer_path_validator(cls, value: Optional[str]):
        """
        Validates the vectorizer path to ensure it exists and is properly formatted.

        Args:
            value: The path to validate

        Returns:
            The validated path

        Raises:
            Validation error if the path is invalid
        """
        return cls.path_validator(value, "vectorizer")

    @field_validator("sequences_path", mode="before")
    def sequences_path_validator(cls, value: Optional[str]):
        """
        Validates the sequences path to ensure it exists and is properly formatted.

        Args:
            value: The path to validate

        Returns:
            The validated path

        Raises:
            Validation error if the path is invalid
        """
        return cls.path_validator(value, "sequences")

    @field_validator("vectorizer_technique", mode="before")
    def vectorize_technique_validator(cls, value: Optional[str]):
        """
        Validates and converts the vectorization technique string to the appropriate enum value.

        Args:
            value: The vectorization technique as a string

        Returns:
            The corresponding VectorizeTechnique enum value

        Raises:
            AIXInvalidVectorTypeError: If the vectorization technique is not recognized
        """
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
        """
        Parses and validates the vectorization hyperparameters.

        Args:
            value: The hyperparameters as a string or other format

        Returns:
            The parsed hyperparameters
        """
        return cls.parse_parameters(value)

    @field_validator("model_hyperparameters", mode="before")
    def model_hyperparameters_validator(cls, value: Optional[str]):
        """
        Parses and validates the model hyperparameters.

        Args:
            value: The hyperparameters as a string or other format

        Returns:
            The parsed hyperparameters
        """
        return cls.parse_parameters(value)

    def transform(self, x):
        """
        Transforms input data using the appropriate vectorization technique.

        This method applies the vectorizer to the input data based on the configured
        vectorization technique.

        Args:
            x: The input data to transform

        Returns:
            The transformed data ready for model input

        Raises:
            AIXVectorizerNotFoundError: If the vectorizer is not instantiated
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
        Makes predictions using the model on the provided input data.

        This method transforms the input data using the appropriate vectorization technique
        and then passes it to the model for prediction.

        Args:
            x: The input data to make predictions on

        Returns:
            The model's predictions as a flattened array, or None if the model is not loaded
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
        """
        Builds and initializes the model and vectorizer from saved files.

        This method loads the model and vectorizer from their respective paths,
        and then loads the sequences data. It sets the model_builded flag to True
        upon successful completion.

        Raises:
            AIXModelNotFoundError: If the model file does not exist
            AIXVectorizerNotFoundError: If the vectorizer file does not exist
        """
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
        """
        Loads and processes sequence data from the specified file.

        This private method loads sequences from either .npy or .pkl files,
        processes them according to the vectorization technique, and applies
        any necessary transformations like sequence reversal. It also cleans
        empty sequences from the loaded data.

        Raises:
            AIXSequencesError: If the file format is unsupported or if the sequences are invalid
        """
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

        if self.clean_empty_sequences:
            self._clean_empty_sequences()

    def _clean_empty_sequences(self):
        """
        Removes empty or invalid sequences from the loaded data.

        This private method identifies and filters out sequences that are empty
        (all zeros), have no elements, or contain only a single non-zero element.
        It updates the sequences attribute with only the valid sequences.

        Raises:
            AIXSequencesError: If no valid sequences remain after cleaning
        """
        if self.sequences is None or len(self.sequences) == 0:
            return

        vectorized_sequences = self.vectorizer(self.sequences)
        valid_indices = []
        for i, record in enumerate(vectorized_sequences):
            if not (np.all(record == 0) or len(record) == 0 or np.sum(record != 0) == 1):
                valid_indices.append(i)

        if len(valid_indices) == 0:
            raise AIXSequencesError("No quedan secuencias válidas después de la limpieza")

        self.sequences = self.sequences[valid_indices]

    def _reverse_sequences(self, sequences):
        """
        Reverses the encoding of sequences back to their original text form.

        This private method converts encoded sequences (indices) back to their
        original text representation using the vectorizer's vocabulary.

        Args:
            sequences: The encoded sequences to reverse

        Returns:
            An array of reversed sequences as text strings
        """
        if not hasattr(self.vectorizer, "get_vocabulary"):
            return sequences

        vocab = self.vectorizer.get_vocabulary()
        reversed_sequences = []

        for seq in sequences:
            words = [vocab[idx] for idx in seq if idx > 0 and idx < len(vocab)]
            reversed_sequences.append(" ".join(words))

        return np.array(reversed_sequences)
