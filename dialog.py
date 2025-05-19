
from rag_processor import *
import json

# Основной сценарий
if __name__ == "__main__":
    dialog = DBConstructor()
    root = "/home/home/Projects/RAGProcessor/FAISS-900" # /РАБОЧИЕ_ПОРЯДКИ_ОБЩИЕ"

    queries = ["Как составить учетную форму для средств измерений (СИ)? Какие сведения должны содержаться в учетной форме для СИ в ПО Метрология?",
             "Как составить учетную форму для вспомогательного оборудования (ВО)? Какие сведения должны содержаться в учетной форме для ВО в ПО Метрология?"]

    # Папки с индексами без имён файлов, проверенные на наличие и правильность
    index_dirs = [d for d, _, files in os.walk(root) for file in files if file.endswith(".faiss")]

    # Один раз вызвали модель эмбеддингов
    set_embs_result = dialog.set_embeddings(root, True)

    # Если эмбеддинги загрузились, то выполняем всё
    if set_embs_result["success"]:
        faiss_load_results = [] # здесь будет список результатов загрузок баз
        for db_folder in index_dirs:
            faiss_load_results.append(dialog.faiss_loader(db_folder, False))

        # Извлекаю список индексов по условию успешной загрузки
        faiss_indexes = [faiss_load_result["db"] for faiss_load_result in faiss_load_results if faiss_load_result["success"]]

        # Задаю префикс E5, если модель эмбеддингов E5 серии
        query_prefix = "query: " if set_embs_result["is_e5_model"] else ""

        # Прохожу по списку вопросов. Это тестовый проход. В реале цикла не будет. Вопросы задает пользователь по одному.
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
