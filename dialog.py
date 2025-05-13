import os
from rag_processor import *
import json
from pprint import pprint

# 1. Список вопросов
questions = [
    "Какой документ регламентирует общие положения учета оборудования?",
    "Кто несет ответственность за обеспечение соответствия оборудования требованиям нормативных актов?",
    "Назовите основные объекты учета, указанные в данном рабочем порядке.",
    "Где хранятся учетные формы оборудования?",
    "Какие категории сотрудников уполномочены вести учетные формы?",
    "Какие обязательные реквизиты должны присутствовать в учетной форме средств измерений (СИ)?",
    "Какая информация фиксируется в журнале учета аттестованных эталонов?",
    "Какие действия предусмотрены при изменении регистрационного номера эталона?",
    "Какие требования предъявляются к оформлению материалов первичной аттестации?",
    "Кто имеет право вносить изменения в учетные формы в ПО Метрология 2.0?",
    "Как обеспечивается защита информации о поверках и калибровках оборудования?",
    "Какие обязательства возлагаются на руководителя подразделения в отношении сохранности учетных данных?",
    "Какие мероприятия проводятся при выявлении неисправностей или повреждений оборудования?",
    "Какие данные фиксируются при прохождении периодической аттестации?",
    "Как регулируется процедура замены устаревшего оборудования новым?",
    "Какие документы подтверждают полномочия лиц, ведущих учет оборудования?",
    "Какие меры принимаются для резервного копирования учетных данных?",
    "Сколько категорий оборудования перечислено в рабочем порядке?"
]


# 2. Утилита для поиска и сбора результатов
def search_and_collect(dialog, questions, faiss_folder):
    results = []
    db_faiss = dialog.faiss_loader(faiss_folder)

    for idx, query in enumerate(questions):
        # Выполняем поиск по каждому вопросу
        output = db_faiss["db"].similarity_search_with_relevance_scores(f"query: {query}", k=5)
        result_list = [{
            "question": query,
            "content": doc.page_content,
            "score": float(round(score, 6)),
            "metadata": doc.metadata
        } for doc, score in output]
        results.append(result_list)

    return results


# 3. Основной сценарий
if __name__ == "__main__":
    dialog = DBConstructor()
    faiss_folder = "/home/home/Projects/RAGProcessor/FAISS/РАБОЧИЕ_ПОРЯДКИ_ОБЩИЕ/_ПР-15-2023_учета_оборудования_и_его_состояния"

    # Собираем результаты поиска
    search_results = search_and_collect(dialog, questions, faiss_folder)

    # Выводим результаты поиска в формате JSON
    json_output = json.dumps(search_results, ensure_ascii=False, indent=4)
    print(json_output)

    # Также можно записать в файл
    with open("search_results.json", "w", encoding="utf-8") as f:
        f.write(json_output)








'''import os
from rag_processor import *

dialog = DBConstructor()

query = "Как составить учетную форму для эталонов средств измерений?"

faiss_folder = "/home/home/Projects/RAGProcessor/FAISS/РАБОЧИЕ_ПОРЯДКИ_ОБЩИЕ/_ПР-15-2023_учета_оборудования_и_его_состояния"
db_faiss = dialog.faiss_loader(faiss_folder)

output = db_faiss["db"].similarity_search_with_relevance_scores(f"query: {query}", k=5) # dialog.hybrid_search_with_scores(query, db_faiss, k=5)
result = [{
    "content": doc.page_content,
    "score": round(score, 6),
    "metadata": doc.metadata
} for doc, score in output]

pprint(result)'''


'''dialog.api_key = os.environ.get("GIGACHAT_API_KEY", None)
dialog.api_url = os.environ.get("GIGACHAT_URL", None)
dialog.ssl = os.environ.get("GIGACHAT_SSL", None)'''

'''def add_title_to_head_chunk(chunks: List[LangDoc]) -> List[LangDoc]:
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

    return output'''

# results = add_title_to_head_chunk(output)
#
# # Вывод результатов
# for i, doc in enumerate(results):
#     print(doc.page_content.strip())