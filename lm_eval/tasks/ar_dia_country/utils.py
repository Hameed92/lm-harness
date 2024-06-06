import datasets
from lm_eval.filters.extraction import Filter, RegexFilter
import re


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # def _process_doc(doc):
    
    #     subject = doc['id'].split("-")[0]
    #     subject_ar = subtasks_ar[subtasks.index(subject)]
    #     out_doc = {**doc, 'subject_ar': subject_ar}
    #     print(subject_ar)
    #     print(out_doc)
    #     return out_doc

    return dataset

class extraction_filter(Filter):
    def apply(self, resps, docs):
        filtered_resps = []
        for res, doc in zip(resps, docs):
            print('doc ===========================', doc)
            print('res ------------------------------', res)
            regex = re.compile(doc['ar_dial'])
            filtered_resps.append(regex.findall(str(res)))
        return filtered_resps