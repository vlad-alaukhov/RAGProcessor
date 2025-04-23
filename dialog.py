from rag_processor import *

dialog = DBConstructor()

query = "Какими должны быть помещения для лабораторной деятельности?"
faiss_folder = "/home/home/Projects/RAGProcessor/DB_FAISS/ПРАВИЛА_ОРГАНИЗАЦИИ_руководства_по_качеству"


def add_title_to_head_chunk(chunks: List[LangDoc]) -> List[LangDoc]:
    """Добавляет _title в content первого чанка каждой связанной группы"""
    processed_doc_ids = set()
    output = []

    for chunk in chunks:
        # Создаем копию чанка (метаданные остаются оригинальными)
        new_content = chunk.page_content
        metadata = chunk.metadata

        # Если это первый чанк документа и есть _title
        if metadata["doc_id"] not in processed_doc_ids and "_title" in metadata:
            new_content = f"{metadata['_title']}\n{new_content}"
            processed_doc_ids.add(metadata["doc_id"])

        output.append(LangDoc(
            page_content=new_content,
            metadata=metadata
        ))

    return output

output = dialog.sim_search(query, faiss_folder, k=10)

results = add_title_to_head_chunk(output)

# Вывод результатов
for i, doc in enumerate(results):
    print(doc.page_content.strip())