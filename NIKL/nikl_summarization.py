# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import textwrap
import zipfile
import unicodedata

import datasets

_VERSION = datasets.Version("1.0.0", "")

_URL = "https://corpus.korean.go.kr/main.do"

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
# 문서 요약 말뭉치

## 소개
(버전 1.0) 문서에서 추출한 주제문과 문서를 요약한 글로 구성된 말뭉치입니다.


## Usage
```python
from datasets import load_dataset

raw_datasets = load_dataset(
                "nikl_summarization.py", 
                "base",
                cache_dir="huggingface_datasets", 
                data_dir="data",
                ignore_verifications=True,
            )

dataset_train = raw_datasets["train"]

for item in dataset_train:
    print(item)
    exit()
```

## Documentation

[Link](https://rlkujwkk7.toastcdn.net/6/NIKL_SUMMARIZATION(v1.0).pdf)

"""

SUMMARIZATION_FNAME_LIST = [
    "NIKL_SUMMARIZATION(v1.0).zip"
]

NEWSPAPER_FNAME_LIST = [
    "NIKL_NEWSPAPER_v2.0.zip",
    "NIKL_NEWSPAPER(v1.0).zip",
]

def find_file_name(root_dir, fpath_list):

    for fpath in fpath_list:
        rel_path = os.path.join(root_dir, fpath)
        if os.path.isfile(rel_path):
            return rel_path
    return None


def _sentences2sentence(sentences):
    return ' '.join([x.strip() for x in sentences])

def _find_fname_from_doc_dict(doc_id, doc_dict):
    doc_key = doc_id.split('.')[0]
    return doc_dict.get(doc_key, None)

def _is_punctuation(char):
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _page_proc(obj):
    raw_example = []
    for paragraph in obj['paragraph']:
        form = paragraph['form'].strip()
        if len(form) > 0:
            if not _is_punctuation(form[-1]):
                form += '.'
            raw_example.append(form)
    return ' '.join(raw_example)

def _find_id_from_doc_summarization(doc_id, f):
    doc_json = json.loads(f.read())
    for doc in doc_json['document']:
        if doc['id'] == doc_id:
            return _page_proc(doc)
    return None


def generator(fpath, doc_fpath, is_single_sent=False):
    with zipfile.ZipFile(doc_fpath, "r") as doc_fp:
        doc_dict = doc_fp.namelist()
        doc_dict = {os.path.splitext(os.path.basename(k))[0]:k for k in filter(lambda x: x.endswith(".json"), doc_dict)}

        with zipfile.ZipFile(fpath, "r") as fp:
            file_list = fp.namelist()
            file_list = filter(lambda x: x.endswith(".json"), file_list)
            for fname in file_list:
                data = json.load(fp.open(fname, "r"))["data"]
                for obj in data:
                    doc_id = obj['document_id']
                    subclass = obj['subclass']
                    head = obj['head']
                    subhead = obj['subhead']

                    doc_fname = _find_fname_from_doc_dict(doc_id, doc_dict)
                    para = _find_id_from_doc_summarization(doc_id, doc_fp.open(doc_fname, "r"))
                    if para is not None:

                        if is_single_sent:
                            for idx, summary_sentences, topic_sentences in zip(range(len(obj["summary_sentences"])), obj["summary_sentences"], obj["topic_sentences"]):
                                summary_sentences = _sentences2sentence(obj["summary_sentences"])
                                topic_sentences = _sentences2sentence(obj["topic_sentences"])
                                yield {
                                    "document_id": doc_id+str(idx),
                                    "subclass": subclass,
                                    "head": head,
                                    "subhead": subhead,
                                    "article": para,
                                    "summary_sentences": summary_sentences,
                                    "topic_sentences": topic_sentences,
                                }

                        else:
                            summary_sentences = _sentences2sentence(obj["summary_sentences"])
                            topic_sentences = _sentences2sentence(obj["topic_sentences"])
                            yield {
                                "document_id": doc_id,
                                "subclass": subclass,
                                "head": head,
                                "subhead": subhead,
                                "article": para,
                                "summary_sentences": summary_sentences,
                                "topic_sentences": topic_sentences,
                            }

class NIKLSummarizationConfig(datasets.BuilderConfig):
    """BuilderConfig for NIKLSummarizationConfig."""

    def __init__(self, **kwargs):
        """BuilderConfig for NIKLSummarizationConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NIKLSummarizationConfig, self).__init__(**kwargs)

        
class NIKLSummarization(datasets.GeneratorBasedBuilder):
    """NIKLSummarization Dataset"""

    BUILDER_CONFIGS = [
        NIKLSummarizationConfig(
            name="base",
            version=datasets.Version("1.0.0"),
            description="NIKL Summarization dataset, concat 3 lines",
        ),
        NIKLSummarizationConfig(
            name="single",
            version=datasets.Version("1.0.0"),
            description="NIKL Summarization dataset, single sentence",
        ),
    ]

    DEFAULT_CONFIG_NAME = "base"

    manual_download_instructions = textwrap.dedent(f"""
    You need to manually download the data file on NIKL (국립국어원 모두의 말뭉치) (${_URL}). 
    The folder containing the saved file can be used to load the dataset 
    via 'datasets.load_dataset("nikl_summarization.py", data_dir="<path/to/folder>")'
    """)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "subclass": datasets.Value("string"),
                    "head": datasets.Value("string"),
                    "subhead": datasets.Value("string"),
                    "article": datasets.Value("string"),
                    "summary_sentences": datasets.Value("string"),
                    "topic_sentences": datasets.Value("string"),
                }
            ),
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        summarization_fpath = find_file_name(dl_manager.manual_dir, SUMMARIZATION_FNAME_LIST)
        newspaper_fpath = find_file_name(dl_manager.manual_dir, NEWSPAPER_FNAME_LIST)

        if summarization_fpath is None:
            raise ValueError(f"Can't find summarization file({SUMMARIZATION_FNAME_LIST}) in {dl_manager.manual_dir}.")
        elif newspaper_fpath is None:
            raise ValueError(f"Can't find newspaper files({NEWSPAPER_FNAME_LIST}) in {dl_manager.manual_dir}.")

        path_kv = {
            datasets.Split.TRAIN: (summarization_fpath, newspaper_fpath),
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath': v1, 'doc_fpath': v2}) for k, (v1, v2) in path_kv.items()
        ]

    def _generate_examples(self, fpath, doc_fpath):
        """Yields examples."""
        is_single_sent = True if "single" in self.config.name else False
        for idx, item in enumerate(generator(fpath, doc_fpath, is_single_sent)):
            yield idx, item

