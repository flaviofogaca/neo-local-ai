from vector_memory import save_memory, search_memory

save_memory(
    "O projeto Neo é um assistente local com memória, streaming e interface web.",
    {"type": "project"}
)

results = search_memory("Que projeto estamos criando?", limit=3)

print(results["documents"])