import datasets
from lm_eval.filters.extraction import Filter
import re
from lm_eval.api.registry import register_filter


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # def _process_doc(doc):
    
    #     subject = doc['id'].split("-")[0]
    #     subject_ar = subtasks_ar[subtasks.index(subject)]
    #     out_doc = {**doc, 'subject_ar': subject_ar}
    #     print(subject_ar)
    #     print(out_doc)
    #     return out_doc

    return dataset

