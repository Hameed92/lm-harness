# coding=utf-8
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
# Lint as: python3
"""The Arabic mmlu from aceGPT benchmark."""

import csv
import os
import textwrap

import pandas as pd
import numpy as np

import datasets
from pathlib import Path
dirname = os.getcwd()

DATA_ROOT = dirname+"/lm_eval/tasks/ar_dia_country/ar_dia_country/ar_dia_dataset/data/"
print('dirname ===========', dirname)
print('dataroot ============', DATA_ROOT)
_ExamsAr_CITATION = """\
@inproceedings{
    
}
"""

_ExamsAr_DESCRIPTION = """\

"""


class ExamsArConfig(datasets.BuilderConfig):
    """BuilderConfig for ExamsAr."""

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        """BuilderConfig for ExamsArConfig.

        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the csv file
          label_column: `string`, name of the column in the csv file corresponding
            to the label
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(ExamsArConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


class ExamsAr(datasets.GeneratorBasedBuilder):
    """The ExamsAr benchmark."""

    BUILDER_CONFIGS = [
        
        ExamsArConfig(
                name= "saudipedia",
                description=textwrap.dedent("ArabicEval exams Arabic splits."),
                text_features={"sentence": "sentence", "ar_dial": "ar_dial"}, 
                label_column="ar_dial",
                data_url="",
                data_dir="",
                citation=textwrap.dedent(
                    """\
                @inproceedings{???
                } 
                """
                ),
                url="",
            ),
                    
        ]

    
    
    def _info(self):
        features = {"sentence": datasets.Value("string"),
                    "ar_dial": datasets.Value("string")}
        features["idx"] = datasets.Value("int32")
        return datasets.DatasetInfo(
            description=_ExamsAr_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _ExamsAr_CITATION,
        )

    def _split_generators(self, dl_manager):
        list_splits = []
        
        list_splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(DATA_ROOT , "arabic_dialects_country_sample.csv"),
                        "split": 'test',
                    },
                )
            )
        # list_splits.append(
        #         datasets.SplitGenerator(
        #             name=datasets.Split.VALIDATION,
        #             gen_kwargs={
        #                 "data_file": os.path.join(DATA_ROOT,"arabic_dialects_country_val.csv"),
        #                 "split": 'dev',
        #             },
        #         )
        #     )
          
        return list_splits
    
    def _generate_examples(self, data_file, split, mrpc_files=None):
        df = pd.read_csv(data_file)
        for index, row in df.iterrows():
            
            example = {'idx': index, "sentence": row['sentence'], 'ar_dial': row['ar_dial']}
            yield example['idx'], example
