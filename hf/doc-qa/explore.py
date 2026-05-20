from sentence_transformers import SentenceTransformer
from retriever import load_index, retrieve
import config

index, metadata = load_index()
embed_model = SentenceTransformer(config.EMBED_MODEL)

question = "What is the Rio Grande Valley?"
results = retrieve(question, embed_model, index, metadata, top_k=4)
for r in results:
  print(f"Score: {r['score']:.4f}")
  print(r["text"][:300])
  print()

  print(f"Score: {r['score']:.4f}")
  print(r["text"][:300])
  print()
