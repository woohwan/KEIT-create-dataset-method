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

"""korquad dataset."""
import os
import json
import zipfile
import copy
import glob
import textwrap
import functools

import datasets


_VERSION = datasets.Version("1.0.0", "")

_URL = "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577"

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
# 뉴스 기사 기계독해 데이터

## 소개
국내 종합일간지 및 지역신문의 뉴스기사를 지문으로 활용, 자연어 질의 응답으로 이루어진 인공지능 학습 데이터
## 구축목적
국내 언론사(중앙일보 등 종합일간지 및 지방지)의 뉴스기사를 지문으로 활용하여 4가지 유형의 질문-답변 세트를 생성, 인공지능을 훈련하기 위한 데이터셋


## Usage
```python
from datasets import load_dataset

raw_datasets = load_dataset(
                "aihub_news_mrc.py", 
                cache_dir="huggingface_datasets", 
                data_dir="data",
                ignore_verifications=True,
            )

dataset_train = raw_datasets["train"]

for item in dataset_train:
    print(item)
    exit()
```

## 데이터 관련 문의처
| 담당자명 | 전화번호 | 이메일 |
| ------------- | ------------- | ------------- |
| 김민경 | 02-6952-9201 | mkgenie@42maru.ai |

## Copyright

### 데이터 소개
AI 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI 응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.
본 AI데이터 등은 인공지능 기술 및 제품·서비스 발전을 위하여 구축하였으며, 지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다.

### 데이터 이용정책
- 본 AI데이터 등을 이용하기 위해서 다음 사항에 동의하며 준수해야 함을 고지합니다.

1. 본 AI데이터 등을 이용할 때에는 반드시 한국지능정보사회진흥원의 사업결과임을 밝혀야 하며, 본 AI데이터 등을 이용한 2차적 저작물에도 동일하게 밝혀야 합니다.
2. 국외에 소재하는 법인, 단체 또는 개인이 AI데이터 등을 이용하기 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가 필요합니다.
3. 본 AI데이터 등의 국외 반출을 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가 필요합니다.
4. 본 AI데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 한국지능정보사회진흥원은 AI데이터 등의 이용의 목적이나 방법, 내용 등이 위법하거나 부적합하다고 판단될 경우 제공을 거부할 수 있으며, 이미 제공한 경우 이용의 중지와 AI 데이터 등의 환수, 폐기 등을 요구할 수 있습니다.
5. 제공 받은 AI데이터 등을 수행기관 등과 한국지능정보사회진흥원의 승인을 받지 않은 다른 법인, 단체 또는 개인에게 열람하게 하거나 제공, 양도, 대여, 판매하여서는 안됩니다.
6. AI데이터 등에 대해서 제 4항에 따른 목적 외 이용, 제5항에 따른 무단 열람, 제공, 양도, 대여, 판매 등의 결과로 인하여 발생하는 모든 민・형사 상의 책임은 AI데이터 등을 이용한 법인, 단체 또는 개인에게 있습니다.
7. 이용자는 AI 허브 제공 데이터셋 내에 개인정보 등이 포함된 것이 발견된 경우, 즉시 AI 허브에 해당 사실을 신고하고 다운로드 받은 데이터셋을 삭제하여야 합니다.
8. AI 허브로부터 제공받은 비식별 정보(재현정보 포함)를 인공지능 서비스 개발 등의 목적으로 안전하게 이용하여야 하며, 이를 이용해서 개인을 재식별하기 위한 어떠한 행위도 하여서는 안됩니다.
9. 향후 한국지능정보사회진흥원에서 활용사례・성과 등에 관한 실태조사를 수행 할 경우 이에 성실하게 임하여야 합니다.

### 데이터 다운로드 신청방법
1. AI 허브를 통해 제공 중인 AI데이터 등을 다운로드 받기 위해서는 별도의 신청자 본인 확인과 정보 제공, 목적을 밝히는 절차가 필요합니다.
2. AI데이터를 제외한 데이터 설명, 저작 도구 등은 별도의 신청 절차나 로그인 없이 이용이 가능합니다.
3. 한국지능정보사회진흥원이 권리자가 아닌 AI데이터 등은 해당 기관의 이용정책과 다운로드 절차를 따라야 하며 이는 AI 허브와 관련이 없음을 알려 드립니다.

"""

# TRAINING_SPANEXT_FPATH_REL: span extraction

TRAINING_SPANEXT_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/1.Training/라벨링데이터/TL_span_extraction.zip"
TRAINING_SPANINFER_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/1.Training/라벨링데이터/TL_span_inference.zip"
TRAINING_ENTAIL_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/1.Training/라벨링데이터/TL_text_entailment.zip"
TRAINING_UNANS_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/1.Training/라벨링데이터/TL_unanswerable.zip"

VALIDATION_SPANEXT_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/2.Validation/라벨링데이터/VL_span_extraction.zip"
VALIDATION_SPANINFER_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/2.Validation/라벨링데이터/VL_span_inference.zip"
VALIDATION_ENTAIL_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/2.Validation/라벨링데이터/VL_text_entailment.zip"
VALIDATION_UNANS_FPATH_REL = "017.뉴스 기사 기계독해 데이터/01.데이터/2.Validation/라벨링데이터/VL_unanswerable.zip"

# extractive, answerable
TRAINING_SQUAD_V1_LIKE_FPATH_REL = [TRAINING_SPANEXT_FPATH_REL]
VALIDATION_SQUAD_V1_LIKE_FPATH_REL = [VALIDATION_SPANEXT_FPATH_REL]

# extractive, answerable + unanswerable
TRAINING_SQUAD_V2_LIKE_FPATH_REL = [TRAINING_SPANEXT_FPATH_REL, TRAINING_UNANS_FPATH_REL]
VALIDATION_SQUAD_V2_LIKE_FPATH_REL = [VALIDATION_SPANEXT_FPATH_REL, VALIDATION_UNANS_FPATH_REL]

# inference, answerable + unanswerable
TRAINING_FPATH_REL = [TRAINING_SPANEXT_FPATH_REL, TRAINING_SPANINFER_FPATH_REL, TRAINING_ENTAIL_FPATH_REL, TRAINING_UNANS_FPATH_REL]
VALIDATION_FPATH_REL = [VALIDATION_SPANEXT_FPATH_REL, VALIDATION_SPANINFER_FPATH_REL, VALIDATION_ENTAIL_FPATH_REL, VALIDATION_UNANS_FPATH_REL]

SQUAD_V1_LIKE_FEATURES = datasets.Features({
    "id":
        datasets.Value("string"),
    "title":
        datasets.Value("string"),
    "context":
        datasets.Value("string"),
    "question":
        datasets.Value("string"),
    "answers":
        datasets.Sequence({
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
        }),
})

SQUAD_V2_LIKE_FEATURES = datasets.Features({
    "id":
        datasets.Value("string"),
    "title":
        datasets.Value("string"),
    "context":
        datasets.Value("string"),
    "question":
        datasets.Value("string"),
    "answers":
        datasets.Sequence({
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
        }),
    "plausible_answers":
        datasets.Sequence({
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
        }),
    "is_impossible": datasets.Value("bool"),
})

INFERENCE_FEATURES = datasets.Features({
    "id":
        datasets.Value("string"),
    "title":
        datasets.Value("string"),
    "context":
        datasets.Value("string"),
    "question":
        datasets.Value("string"),
    "answers":
        datasets.Sequence({
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
            "clue_text": datasets.Value("string"),
            "clue_start": datasets.Value("int32"),
            "options": datasets.Sequence(datasets.Value("string")),
        }),
    "plausible_answers":
        datasets.Sequence({
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
        }),
    "is_impossible": datasets.Value("bool"),
})

YESNO_FEATURES = datasets.Features({
    "id":
        datasets.Value("string"),
    "title":
        datasets.Value("string"),
    "context":
        datasets.Value("string"),
    "question":
        datasets.Value("string"),
    "answers":
        datasets.Sequence({
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
            "clue_text": datasets.Value("string"),
            "clue_start": datasets.Value("int32"),
            "options": datasets.Sequence(datasets.Value("string")),
        }),
})

# fix json error for 'TL_span_inference.zip'. 
# we add '\",' at position 8547047 in 'TL_span_inference.zip'
def load_json(fstring):
    try:
        in_json = json.loads(fstring)
        return in_json
    except json.decoder.JSONDecodeError as e:
        if e.pos == 8547047:
            fstring=fstring[:e.pos] + '\",' + fstring[e.pos:]
            return load_json(fstring)
        else:
            raise e

# adopted from question_answering in tensorflow_datasets 
def generate_squadlike_examples(file_list, is_v2=True, with_clue=False):
    """Parses a SQuAD-like JSON, yielding examples with `SQUAD_LIKE_FEATURES`."""
    # We first re-group the answers, which may be flattened (e.g., by XTREME).
    qas = {}
    
    for filepath in file_list:
        with zipfile.ZipFile(filepath, "r") as fp:
            flist = fp.namelist()
            flist = filter(lambda x: x.endswith(".json"), flist)

            for fname in flist:
                # load string
                fstring = fp.open(fname, "r").read().decode('utf-8')
                mrc_data = load_json(fstring)

                for article in mrc_data["data"]:
                    title = article.get("doc_title", "")
                    for paragraph in article["paragraphs"]:
                        context = paragraph["context"]
                        for qa in paragraph["qas"]:

                            id_ = qa["question_id"]
                            is_impossible = qa.get("is_impossible", False)
                            
                            ans = {
                                "answer_start": qa["answers"]["answer_start"],
                                "text": qa["answers"]["text"],
                                "clue_start": qa["answers"]["clue_start"],
                                "clue_text": qa["answers"]["clue_text"],
                                "options": qa["answers"]["options"],
                            }
                            question = qa["question"]

                            if id_ in qas:
                                if is_impossible:
                                    qas[id_]["plausible_answers"].append(ans)
                                else:
                                    qas[id_]["answers"].append(ans)
                            else:
                                plausible_answers = []
                                answers = []
                                if is_impossible:
                                    plausible_answers = [ans]
                                else:
                                    answers = [ans]

                                qas[id_] = {
                                    "title":title,
                                    "context":context,
                                    "question": question,
                                    "id": id_,
                                    "plausible_answers":plausible_answers,
                                    "answers":answers,
                                    "is_impossible":is_impossible,
                                }

    for id_, qa in qas.items():
        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
        answers = [answer["text"] for answer in qa["answers"]]
        clue_text = [answer["clue_text"] for answer in qa["answers"]]
        clue_start = [answer["clue_start"] for answer in qa["answers"]]
        options = [answer["options"] for answer in qa["answers"]]
        plausible_answer_starts = [answer["answer_start"] for answer in qa["plausible_answers"]]
        plausible_answers = [answer["text"] for answer in qa["plausible_answers"]]

        item = {
                "title": qa["title"],
                "context": qa["context"],
                "question": qa["question"],
                "id": id_,
                "answers": {
                    "answer_start": answer_starts,
                    "text": answers,
                },
            }

        if is_v2:
            item["plausible_answers"] = {
                "answer_start": plausible_answer_starts,
                "text": plausible_answers,
            }
            item["is_impossible"] = qa["is_impossible"]
        
        if with_clue:
            item["answers"]["clue_text"] = clue_text
            item["answers"]["clue_start"] = clue_start
            item["answers"]["options"] = options

        yield item


class AIHubNewsMRCConfig(datasets.BuilderConfig):
    def __init__( self,
                    name='squad.v1.like',
                    training_files=TRAINING_SQUAD_V1_LIKE_FPATH_REL,
                    validation_files=VALIDATION_SQUAD_V1_LIKE_FPATH_REL,
                    features=SQUAD_V1_LIKE_FEATURES,
                    **kwargs):
        super(AIHubNewsMRCConfig, self).__init__(
            name=name,
            version=_VERSION,
            **kwargs
            )
        self.training_files = training_files
        self.validation_files = validation_files
        self.features = features


class AIHubNewsMRCDataset(datasets.GeneratorBasedBuilder):
    """DatasetBuilder for AIHubNewsMRCDataset dataset."""

    BUILDER_CONFIGS = [
        AIHubNewsMRCConfig(
            'squad.v1.like',
            training_files=TRAINING_SQUAD_V1_LIKE_FPATH_REL,
            validation_files=VALIDATION_SQUAD_V1_LIKE_FPATH_REL,
            features=SQUAD_V1_LIKE_FEATURES,
        ),
        AIHubNewsMRCConfig(
            'squad.v2.like',
            training_files=TRAINING_SQUAD_V2_LIKE_FPATH_REL,
            validation_files=VALIDATION_SQUAD_V2_LIKE_FPATH_REL,
            features=SQUAD_V2_LIKE_FEATURES,
        ),
        AIHubNewsMRCConfig(
            'all',
            training_files=TRAINING_FPATH_REL,
            validation_files=VALIDATION_FPATH_REL,
            features=INFERENCE_FEATURES,
        ),
        AIHubNewsMRCConfig(
            'yes_no',
            training_files=[TRAINING_ENTAIL_FPATH_REL],
            validation_files=[VALIDATION_ENTAIL_FPATH_REL],
            features=YESNO_FEATURES,
        ),
    ]

    BUILDER_CONFIG_CLASS = AIHubNewsMRCConfig
    DEFAULT_CONFIG_NAME = "squad.v1.like"

    manual_download_instructions = textwrap.dedent(f"""
        You need to manually download the data file on AIHub (${_URL}). 
        The folder containing the saved file can be used to load the dataset 
        via 'datasets.load_dataset("aihub_news_mrc.py", data_dir="<path/to/folder>")'
    """)

    def _info(self) -> datasets.DatasetInfo:
        """Returns the dataset metadata."""
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""

        path_kv = {
            datasets.Split.TRAIN: [os.path.join(dl_manager.manual_dir, path) for path in self.config.training_files],
            datasets.Split.VALIDATION: [os.path.join(dl_manager.manual_dir, path) for path in self.config.validation_files],
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath_list': v}) for k, v in path_kv.items()
        ]

    def _generate_examples(self, fpath_list):
        """Yields examples."""
        # TODO: Yields (key, example) tuples from the dataset
        if self.config.name.startswith("squad.v1"):
            generator = functools.partial(generate_squadlike_examples, is_v2=False)
        elif self.config.name.startswith("squad.v2"):
            generator = functools.partial(generate_squadlike_examples, is_v2=True)
        elif self.config.name.startswith("all"):
            generator = functools.partial(generate_squadlike_examples, is_v2=True, with_clue=True)
        elif self.config.name.startswith("yes_no"):
            generator = functools.partial(generate_squadlike_examples, is_v2=False, with_clue=True)
        else:
            raise ValueError(f"{self.config.name} doen't exist in the supported config list.")
        
        for idx, item in enumerate(generator(fpath_list)):
            yield idx, item