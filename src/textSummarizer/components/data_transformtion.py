import os
from pathlib import Path

from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
from src.textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        # dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum = "/Users/ameyagidh/AAAmeya/AllWork/Projects/ML/projects/TextSummarization/research/summarizer-data/samsum_dataset"

        # Provide the path to your dataset directory
        dataset_directory = "/Users/ameyagidh/AAAmeya/AllWork/Projects/ML/projects/TextSummarization/research/summarizer-data/samsum_dataset"

        # List the files in the dataset directory
        files = os.listdir(dataset_directory)

        # Create empty lists to store your data
        texts = []
        summaries = []

        # Loop through the files and read the data
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(dataset_directory, file), "r") as f:
                    # Assuming that the file format contains text and summaries
                    text, summary = f.read().split("\n")
                    texts.append(text)
                    summaries.append(summary)

        # Create a dictionary with the data
        data = {
            "text": texts,
            "summary": summaries
        }

        # Create a Dataset object
        dataset = Dataset.from_dict(data)
        dataset_samsum = dataset
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))