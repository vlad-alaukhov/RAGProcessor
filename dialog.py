from rag_processor import *
from typing import List

dialog = DBConstructor()


# Пример использования
query = "Какие требования  проводятся работы по калибровке средств измерений?"
faiss_folder = "/home/home/Projects/RAGProcessor/DB_FAISS/ПРАВИЛА_ОРГАНИЗАЦИИ_руководства_по_качеству"

results = dialog.mmr_search(query, faiss_folder, k=7)

# Вывод результатов
for i, doc in enumerate(results, 1):
    '''print(f"\nРезультат #{i}:")
    print(f"Тип: {doc.metadata['element_type']}")
    print(f"ID: {doc.metadata['chunk_id']}")
    print(f"Связанные чанки: {doc.metadata.get('linked', [])}")
    print("--- Содержание ---")'''
    print(doc.page_content.strip())
    print("-----------------")