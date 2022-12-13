import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter


class AI:
    def __init__(
            self,
            testing_dataset: pd.DataFrame,
            training_dataset: pd.DataFrame,
            number_of_epochs: int,
            activation_function: str,
            MAX_LENGTH: int,

            padding: str,
            truncation: str,

            sequential_data: dict,
            metrics: list,

            sequential_layers=None,
            **kwargs: dict
    ):
        self.training: pd.DataFrame = training_dataset
        self.testing: pd.DataFrame = testing_dataset

        self.training_formatted: list = [None, None]
        self.testing_formatted: list = [None, None]

        self.unique_words_count: int = self.unique_words()

        self.number_of_epochs: int = number_of_epochs
        self.activation_function: str = activation_function
        self.MAX_LENGTH: int = MAX_LENGTH

        self.pad: str = padding
        self.trunc: str = truncation

        self.sequential_data: dict = sequential_data
        self.sequential_layers: dict = sequential_layers if sequential_layers is not None else []
        self.metrics: list = metrics

        self.kwargs: dict = kwargs

        self.optimizer = keras.optimizers.Adam(learning_rate=kwargs.get('learning_rate', 0))

        self.fill_sequence_data()

        self.clean_training_data()
        self.padding()
        self.Sequentiality: keras.models.Sequential = self.Build_Sequentiality()
        self.compiler()

        self.the_library_of_alexandria: list[keras.callbacks.History] = [None, None]

    def train(self):
        self.the_library_of_alexandria[0] = self.Sequentiality.fit(
            self.training_formatted[0],
            self.training_formatted[1],
            epochs=self.number_of_epochs
        )

    def test(self):
        self.the_library_of_alexandria[1] = self.Sequentiality.evaluate(
            self.testing_formatted[0],
            self.testing_formatted[1],
        )

    def clean_training_data(self):
        self.training['text'] = self.training.text.map(self.remove_links)
        self.training['text'] = self.training.text.map(self.remove_punctuation)

        self.testing['text'] = self.testing.text.map(self.remove_links)
        self.testing['text'] = self.testing.text.map(self.remove_punctuation)

        self.training_formatted[0] = self.training['text'].to_numpy()
        self.training_formatted[1] = self.training['target'].to_numpy()
        self.testing_formatted[0] = self.training['text'].to_numpy()
        self.testing_formatted[1] = self.training['target'].to_numpy()
        pass

    @staticmethod
    def tokenizing(tokenizer, inp):
        tokenizer.fit_on_texts(inp[0])
        return tokenizer.texts_to_sequences(inp[0])

    def padding(self):
        tokenizer = Tokenizer(num_words=self.unique_words_count)
        self.training_formatted[0] = pad_sequences(
            self.tokenizing(tokenizer, self.training_formatted),
            maxlen=self.MAX_LENGTH,
            padding=self.pad,
            truncating=self.trunc
        )

        self.testing_formatted[0] = pad_sequences(
            self.tokenizing(tokenizer, self.testing_formatted),
            maxlen=self.MAX_LENGTH,
            padding=self.pad,
            truncating=self.trunc
        )

    @staticmethod
    def remove_links(passed_string: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", "", passed_string)

    @staticmethod
    def remove_punctuation(passed_string: str) -> str:
        return re.sub(r'!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\\|]|^|_|`|{|\||}|~', "", passed_string)

    def unique_words(self) -> int:
        dictionary: dict = {}
        for word in ' '.join(self.training['text']).split():
            dictionary[word] = dictionary.get(word, 0) + 1
        return len(dictionary)

    def fill_sequence_data(self):
        for key in self.sequential_data:
            data: dict = self.sequential_data[key]
            size: int = data.pop("size") if "size" in data else -1
            match key:
                case "embedding":
                    data['input_dim'] = data.get('input_dim', self.unique_words_count)
                    data['output_dim'] = data.get('output_dim', size if size >= 0 else 32)
                    data['input_length'] = data.get("input_length", self.MAX_LENGTH)

                case "lstm":
                    data['units'] = data.get('units', size if size >= 0 else 64)
                    # data['dropout'] = data.get('dropout', self.dropout)

                case "dense":
                    data['units'] = data.get('units', size if size >= 0 else 1)
                    data['activation'] = data.get('activation', self.activation_function)

    def Build_Sequentiality(self) -> keras.models.Sequential:
        sequence_layers: list = []

        for key in self.sequential_data:
            data: dict = self.sequential_data[key]
            size: int = data.pop("size") if "size" in data else -1
            match key:
                case "embedding":
                    sequence_layers.append(layers.Embedding(
                        **data,
                    ))

                case "lstm":
                    sequence_layers.append(layers.LSTM(
                        **data,
                    ))

                case "dense":
                    sequence_layers.append(layers.Dense(
                        **data
                    ))

                case _:
                    sequence_layers.append(self.sequential_layers[key](
                        **data
                    ))

        return keras.models.Sequential(sequence_layers)

    def compiler(self):
        self.Sequentiality.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=self.optimizer,
            metrics=["accuracy"]
        )

    def history(self) -> dict:
        return {
            "training": self.the_library_of_alexandria[0].history if self.the_library_of_alexandria[0] is not None else {},
            'testing': {key: val for key, val in zip(self.Sequentiality.metrics_names, self.the_library_of_alexandria[1] if self.the_library_of_alexandria[0] is not None else [0, 0])}
        }
