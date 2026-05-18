# need to run this command in the terminal to download the spaCy model before running the script
# python -m spacy download en_core_web_sm
# pip install pyvis==0.1.9
import os
import pandas as pd
import spacy
from collections import defaultdict
import networkx as nx
import json
from pyvis.network import Network

# Create the output directory if it doesn't exist
output_dir = "ner_results"
os.makedirs(output_dir, exist_ok=True)

# Load the preprocessed CSV
csv_path = "./data/romeo_juliet_preprocessed.csv"
data = pd.read_csv(csv_path)

# Inspect the data
print(data.head())

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    entities = {"characters": [], "families": [], "locations": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["characters"].append(ent.text)
        elif ent.label_ == "ORG":  # Assuming families are labeled as organizations
            entities["families"].append(ent.text)
        elif ent.label_ == "GPE":  # Geopolitical entities for locations
            entities["locations"].append(ent.text)
    return entities

# Apply NER to the dialogue column
data["entities"] = data["dialogue"].apply(extract_entities)

# Inspect the results
print(data[["dialogue", "entities"]].head())

# Initialize profiles
entity_profiles = defaultdict(lambda: {"mentions": 0, "relationships": set()})

# Build profiles
for _, row in data.iterrows():
    entities = row["entities"]
    for char in entities["characters"]:
        entity_profiles[char]["mentions"] += 1
        entity_profiles[char]["relationships"].update(entities["characters"])

# Convert to a structured format
entity_profiles = {
    k: {"mentions": v["mentions"], "relationships": list(v["relationships"])}
    for k, v in entity_profiles.items()
}

# Inspect profiles
print(entity_profiles)

# Initialize graph
G = nx.Graph()

# Add edges based on relationships
for _, row in data.iterrows():
    entities = row["entities"]["characters"]
    for i, char1 in enumerate(entities):
        for char2 in entities[i+1:]:
            if G.has_edge(char1, char2):
                G[char1][char2]["weight"] += 1
            else:
                G.add_edge(char1, char2, weight=1)

# Save the graph as a GEXF file
graph_path = os.path.join(output_dir, "entity_relationship_graph.gexf")
nx.write_gexf(G, graph_path)

# Save entity profiles as a JSON file
profiles_path = os.path.join(output_dir, "entity_profiles.json")
with open(profiles_path, "w") as f:
    json.dump(entity_profiles, f)

print(f"Entity profiles saved to {profiles_path}")

# Create a Pyvis network
net = Network(notebook=False)

# Add nodes and edges from the NetworkX graph
for node in G.nodes:
    net.add_node(node, label=node)

for edge in G.edges(data=True):
    net.add_edge(edge[0], edge[1], value=edge[2]["weight"])

# Save and open the graph as an HTML file
html_path = os.path.join(output_dir, "entity_relationship_graph.html")
net.show(html_path)

print(f"Graph visualization saved to {html_path}")
