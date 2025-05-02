from pprint import pprint

from rag_processor import *


dialog = DBConstructor()

query = "Сведения учетной формы для СИ должны содержать"
faiss_folder = "/home/home/Projects/RAGProcessor/FAISS/РАБОЧИЕ_ПОРЯДКИ_ОБЩИЕ/ПР-15-2023_учета_оборудования_и_его_состояния"


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

db_faiss = dialog.faiss_loader(faiss_folder, True)

output = dialog.sim_search_with_scores(query, db_faiss, k=10)

pprint(output)

# results = add_title_to_head_chunk(output)
#
# # Вывод результатов
# for i, doc in enumerate(results):
#     print(doc.page_content.strip())