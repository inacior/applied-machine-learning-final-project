import pandas as pd
from pathlib import Path

models = [
    ('Llama 3B',     'llama3b'),
    ('Ministral 3B', 'ministral3b'),
    ('Mistral 7B',   'mistral7b'),
    ('Gemma 3 4B',   'gemma34b'),
    ('Mistral Saba', 'mistral_saba'),
    ('Qwen 2.5 7B',  'qwen25_7b'),
]

for label, folder in models:
    path = Path(f'results/{folder}/answers_before_ner.csv')
    if path.exists():
        df = pd.read_csv(path)
        total = len(df)
        correct = (df['evaluation'] == 'CORRECT').sum()
        partial = (df['evaluation'] == 'PARTIALLY_CORRECT').sum()
        print(f'{label}: strict={correct/total*100:.1f}%, lenient={(correct+partial)/total*100:.1f}%')
    else:
        print(f'{label}: arquivo nao encontrado em {path}')