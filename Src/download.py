import os
import json
import datasets
import gzip
from collections import defaultdict
from dataclasses import dataclass

_CITATION = '''
'''

MIRACL_DATA_DIR = "/Data/miracl"
PLM_DIR = "/PLM"

languages2filesize = {
    'ar': 5, 'bn': 1, 'en': 66, 'es': 21, 'fa': 5, 'fi': 4, 'fr': 30,
    'hi': 2, 'id': 3, 'ja': 14, 'ko': 3, 'ru': 20, 'sw': 1, 'te': 2,
    'th': 2, 'zh': 10, 'de': 32, 'yo': 1,
}


_DATASET_URLS = {
    lang: {
        'train': [
            f'https://hf-mirror.com/datasets/miracl/miracl-corpus/resolve/main/miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz'
            for i in range(n)
        ]
    } for lang, n in languages2filesize.items()
}


PLM_MODELS = [
    "mdpr-tied-pft-msmarco",
    "bge-m3",
    "multilingual-e5-large"
]

class MIRACLCorpus(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=datasets.Version('1.0.0'),
            name=lang,
            description=f'MIRACL dataset in language {lang}.'
        ) for lang in languages2filesize
    ]

    def _info(self):
        features = datasets.Features({
            'docid': datasets.Value('string'),
            'title': datasets.Value('string'),
            'text': datasets.Value('string'),
        })
        return datasets.DatasetInfo(
            description='MIRACL dataset loader.',
            features=features,
            supervised_keys=None,
            homepage='https://project-miracl.github.io',
            license='',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name
        local_dir = os.path.join(MIRACL_DATA_DIR, lang)
        os.makedirs(local_dir, exist_ok=True)
        

        downloaded_files = dl_manager.download_and_extract(_DATASET_URLS[lang])
        return [
            datasets.SplitGenerator(
                name='train',
                gen_kwargs={'filepaths': downloaded_files['train']},
            )
        ]

    def _generate_examples(self, filepaths):
        for filepath in sorted(filepaths):
            with gzip.open(filepath, 'rt', encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    yield data['docid'], data

if not os.path.exists(PLM_DIR):
    os.makedirs(PLM_DIR)
    for model in PLM_MODELS:
        os.system(f'wget -P {PLM_DIR} https://huggingface.co/{model}/resolve/main')
