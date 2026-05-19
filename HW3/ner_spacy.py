import pandas as pd
import spacy
from tqdm import tqdm

# Carregar modelo spaCy
print("Carregando modelo spaCy...")
nlp = spacy.load("en_core_web_lg")

# Carregar o dataset original
print("Carregando dataset...")
df = pd.read_csv("data/romeo_juliet_preprocessed.csv")

# Função que extrai entidades de um texto
def extract_entities(text):
    if not isinstance(text, str) or text.strip() == "":
        return "", ""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entity_texts = " | ".join([e[0] for e in entities])
    entity_labels = " | ".join([e[1] for e in entities])
    return entity_texts, entity_labels

# Aplicar NER em cada linha de diálogo
print("Aplicando NER nas linhas de diálogo (pode demorar alguns minutos)...")
tqdm.pandas()
df[["ner_entities", "ner_labels"]] = df["dialogue"].progress_apply(
    lambda x: pd.Series(extract_entities(x))
)

# Salvar o novo CSV
output_path = "data/thea_ner_augmented.csv"
df.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")
print(f"Total de linhas processadas: {len(df)}")
print("\nExemplo das primeiras linhas:")
print(df[["character", "dialogue", "ner_entities", "ner_labels"]].head(3).to_string())