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
        output = db_faiss["db"].similarity_search_with_relevance_scores(f"query: {query}", k=5, )
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
    root = "/home/home/Projects/RAGProcessor/FAISS/РАБОЧИЕ_ПОРЯДКИ_ОБЩИЕ"

    queries = ["Как составить учетную форму для средств измерений (СИ)? Какие сведения должны содержаться в учетной форме для СИ в ПО Метрология?",
             "Как составить учетную форму для вспомогательного оборудования (ВО)? Какие сведения должны содержаться в учетной форме для ВО?"]

    # Папки с индексами без имён файлов, проверенные на наличие и правильность
    index_dirs = [d for d, _, files in os.walk(root) for file in files if file.endswith(".faiss")]

    # Один раз вызвали модель эмбеддингов
    set_embs_result = dialog.set_embeddings(root, False)

    # Если эмбеддинги загрузились, то выполняем всё
    if set_embs_result["success"]:
        faiss_load_results = [] # здесь будет список результатов загрузок баз
        for db_folder in index_dirs:
            faiss_load_results.append(dialog.faiss_loader(db_folder, False))

        # Извлекаю список индексов по условию успешной загрузки
        faiss_indexes = [faiss_load_result["db"] for faiss_load_result in faiss_load_results if faiss_load_result["success"]]

        # Задаю префикс E5, если модель эмбеддингов E5 серии
        query_prefix = "query: " if set_embs_result["is_e5_model"] else ""

        # Прохожу по списку вопросов. В реале цикла не будет. Это тестовый проход
        for idx, query in enumerate(queries):
            print(query) # Вопрос
            # Получаю разультат поиска по группе баз
            results = dialog.process_query(query_prefix+query,
                                           faiss_indexes,
                                           dialog.aformatted_scored_sim_search_by_cos,
                                           k=5)
            for result in results[:3]: print(result["content"])
            # Выводим результаты поиска в формате JSON
            json_output = json.dumps(results, ensure_ascii=False, indent=4)

            # Также можно записать в файл
            with open(f"search_results_{idx}.json", "w", encoding="utf-8") as f:
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