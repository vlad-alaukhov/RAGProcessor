import json
import shutil
from abc import ABC
import fitz
from camelot import read_pdf
import hashlib
import pandas as pd
from docx import Document as Docx
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter     # рекурсивное разделение текста
from langchain.docstore.document import Document as LangDoc
import tiktoken
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re                 # работа с регулярными выражениями
import requests
from dotenv import load_dotenv
import time
# from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Any, Dict, Generator
from langchain_core.embeddings import Embeddings

class RAG(ABC):
    def __init__(self):
        os.environ.clear()
        load_dotenv(".venv/.env")
        # получим переменные окружения из .env
        self.api_url = os.environ.get("OPENAI_URL")
        # API-key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # HF-токен
        self.hf = os.environ.get("HF-TOKEN")

class RAGProcessor(RAG):
    def __init__(self):
        super().__init__()

    def request_to_openai(self, system: str, request: str, temper: float, model="openai/gpt-4o-mini", verbose=False):
        attempts = 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request}
            ],
            "temperature": temper,
        }

        if verbose:
            print("===============================================")
            print("model: ", model)
            print("-----------------------------------------------")
            print("system: ", system)
            print("-----------------------------------------------")
            print("user: ", request)
            print("-----------------------------------------------")

        while attempts < 3:
            try:
                time.sleep(10)
                # Отправляем POST-запрос к модели
                response = requests.post("https://api.vsegpt.ru:6010/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  # Проверка на наличие ошибок в запросе

                # Получаем ответ от модели
                response_data = response.json()
                result_text = response_data['choices'][0]['message']['content']
                if verbose: print("Ответ модели: ", result_text)
                return True, result_text

            except Exception as e:
                print(e)
                attempts += 1
                if attempts >= 3: return False, f"Ошибка генерации: {e}"


class DBConstructor(RAGProcessor):
    def __init__(self):
        super().__init__()
        self.embeddings = None
        self.db_metadata = None
        self.chunk_size = 700
        self.source_chunks = None
        self.num_tokens = 0
        self.summary = None
        self.db = None
        self.answer = None
        self.unprocessed_text = None
        self.processed_text = None

    @staticmethod
    def pdf_parser(files: str | list):
        """
        Парсит текст из PDF-ок.
        Ищет все PDF-ки в pdf_files, парсит текст в переменную text, Записывает в текстовый base_file
        :param files: Путь/к/файлу.pdf или список файлов
        :return: Кортеж (Code, "Результат") Code: True, если всё в порядкеБ False, если ошибка.
        """
        if files is None or len(files) == 0: return False, "Файл не выбран"
        if type(files) == str: files = [files]
        pdf_files = [fn for fn in files if fn.endswith('.pdf')]

        for each_pdf in pdf_files:
            try:
                with fitz.open(each_pdf) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text()
            except Exception as e:
                return False, f"Ошибка при обработке файла {each_pdf}: {e}"
        if text: return True, text

    @staticmethod
    def minus_words(file_name: str, pattern: str, to_change: str):
        """ Замена слов в файле. Открывает файл, грузит в переменную, находит pattern, меняет на to_change
        бэкапит файл и переписывает измененный текст в исходный файл.
        :param file_name: имя файла, в котором надо заменить слова
        :param pattern: Что менять
        :param to_change: На что менять
        """
        with open(file_name, 'r') as file:
            text = file.read()

        out_text = re.sub(pattern, to_change, text)
        os.rename(file_name, file_name.split('.')[0] + '.bak')
        with open(file_name, 'w') as fn:
            fn.write(out_text)

#=============================================================================
# Парсинг документов pdf, docx, xlsx
    def document_parser(self, file_path: str) -> list:
        """Универсальный парсер документов с единой структурой метаданных"""
        if file_path.endswith('.docx'):
            return self._parse_docx(file_path)
        elif file_path.endswith('.pdf'):
            return self._parse_pdf(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return self._parse_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

    def _parse_docx(self, file_path: str) -> list:

        doc = Docx(file_path)
        # Готовлю ид документа. Хэширую путь файла
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        chunks = []

        # Собираем в список элементы документа вообще все
        elements = [elem for elem in doc.element.body if elem.tag.endswith(('p', 'tbl'))]

        groups = []
        current_group = ""

        for i, elem in enumerate(elements):

            if elem.tag.endswith('p'):
                text = elem.text.strip()

                # Пустой абзац = разделитель группы
                if not text:
                    print(f"Пробел {i}")
                    if current_group:
                        groups.append(current_group)
                        current_group = ""
                    continue

                print(f"Текст {i}")

                current_group += elem.text.strip() + '\n'  # Добавляем НЕпустой абзац

            elif elem.tag.endswith('tbl'):
                # Прерываем текст перед таблицей
                if current_group:
                    groups.append(current_group.strip())
                    current_group = ""

                print(f"Таблица {i}")
                try:
                    # Получаем объект таблицы через API
                    table = next(t for t in doc.tables if t._element is elem)
                    table_data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                    groups.append(table_data)
                except Exception as e:
                    print(f"Ошибка обработки таблицы: {str(e)}")
                    prev_table_id = None
                    continue

            # Добавляем последнюю группу (если не была добавлена)
        if current_group:
            groups.append(current_group)

        for each in groups: print(each)
        return groups

    def _parse_pdf(self, file_path: str) -> list:
        """Парсинг PDF с базовым разделением текста и таблиц"""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        chunks = []

        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                # Текст страницы
                text = page.get_text().strip()
                if text:
                    chunks.append(LangDoc(
                        page_content=text,
                        metadata={
                            "doc_id": doc_id,
                            "doc_type": "pdf",
                            "chunk_id": f"{doc_id}_p{page_num + 1}_text",
                            "element_type": "text",
                            "linked": []
                        }
                    ))

                # Таблицы
                tables = read_pdf(file_path, pages=str(page_num + 1), flavor="stream")
                for i, table in enumerate(tables):
                    chunks.append(LangDoc(
                        page_content=table.df.to_json(),
                        metadata={
                            "doc_id": doc_id,
                            "doc_type": "pdf",
                            "chunk_id": f"{doc_id}_p{page_num + 1}_table{i + 1}",
                            "element_type": "table",
                            "linked": [chunks[-1].metadata["chunk_id"]] if chunks else []
                        }
                    ))

        return chunks

    def _parse_excel(self, file_path: str) -> list:
        """Парсинг Excel с сохранением структуры листов"""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        chunks = []

        dfs = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in dfs.items():
            # Текстовое представление листа
            chunks.append(LangDoc(
                page_content=f"Лист: {sheet_name}\n{df.to_string()}",
                metadata={
                    "doc_id": doc_id,
                    "doc_type": "excel",
                    "chunk_id": f"{doc_id}_{sheet_name}_text",
                    "element_type": "text",
                    "linked": [f"{doc_id}_{sheet_name}_table"]
                }
            ))

            # Табличные данные
            chunks.append(LangDoc(
                page_content=df.to_json(),
                metadata={
                    "doc_id": doc_id,
                    "doc_type": "excel",
                    "chunk_id": f"{doc_id}_{sheet_name}_table",
                    "element_type": "table",
                    "linked": [f"{doc_id}_{sheet_name}_text"]
                }
            ))

        return chunks

# ========================================================
    @staticmethod
    def validate_chunks(chunks: list) -> bool:
        for chunk in chunks:
            for linked_id in chunk.metadata["linked"]:
                if not any(c.metadata["chunk_id"] == linked_id for c in chunks):
                    raise ValueError(f"Битая связь: {chunk.metadata['chunk_id']} → {linked_id}")
        return True

# Подготовка к векторизации сложных документов

    @staticmethod
    def validate_link(chunk: LangDoc, chunks: List[LangDoc]):
        for linked_id in chunk.metadata["linked"]:
            if not any(ch.metadata["chunk_id"] == linked_id for ch in chunks):
                return None
            else:
                return chunk.metadata["linked"]

    def prepare_chunks(self, dry_chunks: list) -> List[LangDoc]:
        processed_chunks = []
        doc_id = None
        prev_text_chunk_id = None
        prev_table_id = None
        prev_was_table = False  # Флаг для отслеживания последовательных таблиц
        global_counter = 0

        for idx, chunk in enumerate(dry_chunks):
            is_table = isinstance(chunk, list) and all(isinstance(row, list) for row in chunk)

            # Инициализация doc_id
            if doc_id is None and not is_table:
                doc_id = hashlib.md5(str(chunk).encode()).hexdigest()[:8]

            # Обработка текста
            if not is_table:
                if len(chunk) > 800:
                    split_texts = self.split_text_recursive(chunk, 800)
                else:
                    split_texts = [chunk]

                prev_sub_id = None
                for i, text in enumerate(split_texts):
                    global_counter += 1
                    metadata = {
                        "doc_id": doc_id,
                        "doc_type": "docx",
                        "chunk_id": f"{doc_id}_{global_counter}",
                        "element_type": "text",
                        "linked": []
                    }

                    # Связь внутри подчанков
                    if prev_sub_id:
                        metadata["linked"].append(prev_sub_id)
                        prev_chunk = next(c for c in processed_chunks if c.metadata["chunk_id"] == prev_sub_id)
                        prev_chunk.metadata["linked"].append(metadata["chunk_id"])

                    processed_chunks.append(LangDoc(page_content=text, metadata=metadata))
                    prev_sub_id = metadata["chunk_id"]
                    prev_text_chunk_id = metadata["chunk_id"]

                prev_was_table = False  # Сброс флага после текста

            # Обработка таблицы
            else:
                global_counter += 1
                metadata = {
                    "doc_id": doc_id,
                    "doc_type": "docx",
                    "chunk_id": f"{doc_id}_{global_counter}",
                    "element_type": "table",
                    "linked": []
                }

                # Связь с предыдущим текстовым чанком
                if prev_text_chunk_id:
                    metadata["linked"].append(prev_text_chunk_id)
                    prev_chunk = next(c for c in processed_chunks if c.metadata["chunk_id"] == prev_text_chunk_id)
                    prev_chunk.metadata["linked"].append(metadata["chunk_id"])

                # Связь с предыдущей таблицей ТОЛЬКО если они идут подряд
                if prev_was_table and prev_table_id:
                    metadata["linked"].append(prev_table_id)
                    prev_table_chunk = next(c for c in processed_chunks if c.metadata["chunk_id"] == prev_table_id)
                    prev_table_chunk.metadata["linked"].append(metadata["chunk_id"])

                processed_chunks.append(LangDoc(page_content=str(chunk), metadata=metadata))
                prev_text_chunk_id = None
                prev_table_id = metadata["chunk_id"]
                prev_was_table = True  # Устанавливаем флаг

        return processed_chunks

#==========================================================================================

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Возвращает количество токенов в строке"""
        encoding = tiktoken.get_encoding(encoding_name)
        self.num_tokens = len(encoding.encode(string))
        return self.num_tokens

    def split_text_recursive(self, text: str, chunk_size: int):
        """
        Делит текст в строковой переменной text методом RecursiveCharacterTextSplitter
        на чанки размером chunk_size
        :param text: Текст в строке. Чтобы дополнительно разделить langchain-документ, надо подавать page_content
        :param chunk_size: Размер чанка.
        :return: Список чанков типа str
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n'],
            chunk_size=chunk_size,
            chunk_overlap=6
        )

        self.source_chunks = splitter.split_text(text)
        return self.source_chunks

    def split_recursive_from_markdown(self, documents_to_split: list, chunk_size: int, verbose=False) -> list:
        """ Делит список Langchain документов documents_to_split на чанки размером chunk_size
        методом RecursiveCharacterTextSplitter. Для этого вызывается метод split_text_recursive,
        реализованный в этом же классе.
        :param documents_to_split: Текст базы
        :param chunk_size: Размер чанка
        :return: список Langchain документов
        """
        source_chunks = []
        if verbose: print(f"Поступило чанков для разбиения: {len(documents_to_split)}")

        for each_document in documents_to_split:
            header = ''
            new_chunks = self.split_text_recursive(each_document.page_content, chunk_size)
            if verbose: print(f"Чанков после разбиения: {len(new_chunks)}")
            for each_element in new_chunks:
                if len(each_element) > 1:
                    for key in each_document.metadata:
                        header += f"{each_document.metadata[key]}. "
                source_chunks.append(LangDoc(page_content=header+each_element, metadata=each_document.metadata))
        if verbose: print(f"Чанков всего: {len(source_chunks)}")

        return source_chunks

    @staticmethod
    def split_markdown(db_text: str):
    # MarkDownHeader разметка базы из файла с дублированием заголовка в текст чанка

        hd_level = len(max(re.findall(r'#+', db_text)))
        print(hd_level)

        headers_to_split_on = [(f"{'#' * n}", f"H{n}") for n in range(1, hd_level+1 if hd_level >= 1 else 1)]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = markdown_splitter.split_text(db_text)

        return fragments

    # Подсчет токенов
    @staticmethod
    def num_tokens_from_messages(messages, model='gpt-4o-mini'):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')

        if model in ['gpt-4o-mini', 'gpt-4o', 'gpt-4o-latest']:
            num_tokens = 0

            for message in messages:
                num_tokens += 4

                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))

                    if key == 'name':
                        num_tokens -= 1

            num_tokens += 2
            return num_tokens

        else:
            raise NotImplementedError(f'''num_tokens_from_messages() is not presently implemented for model {model}.''')

    def db_pre_constructor(self, text: str, system: str, user: str, chunk_size=0, verbose=False):
        """
        Метод для предварительной обработки базы. Открывается файл неразмеченной базы и размечается по крупным
        разделам путем деления на чанки RecursiveCharacterTextSplitter из метода
        self.split_text_recursive(text, chunk_size).
        Разделы размечаются промптом "Крупные разделы" из файла prompts.yaml
        :param text:
        :param system: Системный промпт system = prompts['Крупные разделы']['system']
        :param user: Юзер-промпт user = prompts['Крупные разделы']['user']
        :param chunk_size: Размер чанков. По умолчанию 10000, чтобы влезли в модель
        :param verbose: True - выводит на печать отладочную информацию
        :return:
        """
        code = None

        # делю на чанки
        if chunk_size != 0:
            self.source_chunks = self.split_text_recursive(text, chunk_size)
            if verbose: print(f"Текст разделен на {len(self.source_chunks)} отрезков.\nОбщая длина текста: {len(text)}")
        else: self.source_chunks = [text]

        result_text = ""

        for num, chunk in enumerate(self.source_chunks):
            request = f"{user}\n{chunk}"
            code, self.answer = self.request_to_openai(system, request, 0)
            if verbose: print(f'Отрезок №{num}, длина: {len(chunk)}\n{self.answer}')  # Выводим ответ
            if code: result_text += self.answer
            else: break

        return code, self.answer

    def db_constructor(self, text: str, system: str, user: str, verbose=False):
        """
        Конструктор базы знаний
        Принимает text из разметки markdown в виде строки, делит на отрезки и формирует запрос и отправляет на ChatGPT.
        Из ChatGPT принимает код ответа и ответ системы. Если код ответа будет False, то выходит из цикла.
        В случае успеха (код True), накапливает результат.
        Возвращает код ответа (True, False) и ответ системы или сообзение об ошибке.
        """
        result_text = ''
        code = None

        fragments = self.split_markdown(text)
        if verbose: print(f"Всего фрагментов: {len(fragments)}")

        for num, fragment in enumerate(fragments):
            if verbose: print(f"Текущий фрагмент №{num+1}:\n{fragment}\n_______________________________________")
            request = f"{user}\n{fragment.page_content}"
            code, answer = self.request_to_openai(system, request, 0)
            if verbose:
                print(f"Обработанный фрагмент:\n{answer}")
                print("____________________________________________________")
                print(f"Код выполнения: {code}; Длина сегмента: {len(fragment.page_content)}")
                print("====================================================")
            if code: result_text += f"{answer}\n\n"
            else: break

        return code, result_text
    # ============================================================================
    # Всё, что касается векторизации

    def vectorizator(self, docs: list, db_folder: str, **kwargs):
        """Универсальный метод векторизации с автонастройкой для E5"""
        try:
            model_type = kwargs.get("model_type", "huggingface").lower()
            model_name = kwargs.get("model_name", "")
            is_e5_model = "e5" in model_name.lower()

            # Валидация параметров
            if not model_name: return False, "Не указано название модели"

            # Автоматические настройки для E5
            encode_kwargs = kwargs.get("encode_kwargs", {})
            model_kwargs = kwargs.get("model_kwargs", {})

            if is_e5_model:
                # Принудительная нормализация и префиксы
                encode_kwargs.update({
                    'normalize_embeddings': True,
                    'batch_size': 64,
                    'convert_to_numpy': True
                })
                # Добавляем префиксы к текстам
                docs = self._add_e5_prefixes(docs)

            # Создаем эмбеддинги
            if model_type == "openai":
                embeddings = OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key,
                    base_url=self.api_url
                )
                distance_strategy = "COSINE"

            elif model_type == "huggingface":
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                distance_strategy = "COSINE" if is_e5_model else "L2"
            else:
                return False, f"Неподдерживаемый тип модели: {model_type}"

            # Создаем и сохраняем индекс
            self.db = FAISS.from_documents(
                documents=docs,
                embedding=embeddings,
                distance_strategy=distance_strategy
            )
            self.db.save_local(db_folder)

            # Сохраняем метаданные с дополнительными параметрами
            metadata = {
                "embedding_model": model_name,
                "model_type": model_type,
                "dimension": self._get_embedding_dimension(embeddings),
                "normalized": encode_kwargs.get('normalize_embeddings', False),
                "distance_strategy": distance_strategy,
                "is_e5_model": is_e5_model
            }
            try:
                with open(os.path.join(db_folder, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except (PermissionError, OSError) as e:
                return False, f"Ошибка записи метаданных: {str(e)}"
            except (TypeError, json.JSONDecodeError) as e:
                return False, f"Ошибка формата метаданных: {str(e)}"

            return True, f"База успешно создана в {db_folder}"

        except Exception as e:
            return False, f"Ошибка векторизации: {str(e)}"

    @staticmethod
    def _add_e5_prefixes(docs):
        """Добавляет E5-префиксы к документам"""
        for doc in docs:
            if doc.page_content.startswith("query:") or doc.page_content.startswith("passage:"):
                continue
            doc.page_content = f"passage: {doc.page_content}"
        return docs

    @staticmethod
    def _get_embedding_dimension(embeddings):
        """Определение размерности с обработкой исключений"""
        try:
            if isinstance(embeddings, OpenAIEmbeddings):
                return len(embeddings.embed_query("test"))
            elif isinstance(embeddings, HuggingFaceEmbeddings):
                return len(embeddings.embed_query("test"))
        except Exception as e:
            print(f"Ошибка определения размерности: {str(e)}")
        return "unknown"

    def hybrid_vectorizator(
            self,
            docs: List[LangDoc],
            db_folder: str,
            text_model: str = "cointegrated/LaBSE-ru-turbo",
            table_model: str = "deepset/all-mpnet-base-v2-table",
            **kwargs
    ) -> tuple:
        """Векторизация с разделением текста и таблиц. Не изменяет старый vectorizator."""
        try:
            # Разделяем чанки
            text_chunks = [d for d in docs if d.metadata.get("element_type") != "table"]
            table_chunks = [d for d in docs if d.metadata.get("element_type") == "table"]

            # Создаем подпапки для текстов и таблиц
            text_db_path = os.path.join(db_folder, "text_db")
            table_db_path = os.path.join(db_folder, "table_db")
            os.makedirs(text_db_path, exist_ok=True)
            os.makedirs(table_db_path, exist_ok=True)

            # Векторизация текстов (используем СТАРЫЙ vectorizator)
            if text_chunks:
                success, msg = self.vectorizator(
                    docs=text_chunks,
                    db_folder=text_db_path,
                    model_name=text_model,
                    model_type="huggingface",
                    **kwargs
                )
                if not success:
                    return False, f"Ошибка текстовой базы: {msg}"

            # Векторизация таблиц (используем СТАРЫЙ vectorizator)
            if table_chunks:
                success, msg = self.vectorizator(
                    docs=table_chunks,
                    db_folder=table_db_path,
                    model_name=table_model,
                    model_type="huggingface",
                    **kwargs
                )
                if not success:
                    return False, f"Ошибка табличной базы: {msg}"

            # Копируем метаданные из text_db в корень для совместимости
            if os.path.exists(os.path.join(text_db_path, "metadata.json")):
                shutil.copy(
                    os.path.join(text_db_path, "metadata.json"),
                    os.path.join(db_folder, "metadata.json")
                )

            return True, f"Гибридные базы сохранены в {db_folder}"
        except Exception as e:
            return False, f"Критическая ошибка: {str(e)}"

    #=======================================================================
    # Загрузка базы

    def faiss_loader(self, db_folder: str, hybrid_mode: bool = False) -> Dict[str, Any]:
        """Загрузка базы с поддержкой гибридного режима."""
        result = {
            "success": False,
            "db": None,
            "text_db": None,  # Только для hybrid_mode
            "table_db": None,  # Только для hybrid_mode
            "error": ""
        }

        try:
            if not hybrid_mode:
                # Старый режим (для обратной совместимости)
                load_result = self._load_single_db(db_folder)
                if not load_result["success"]:
                    raise ValueError(load_result["error"])
                result["db"] = load_result["db"]
            else:
                # Гибридный режим
                text_db_result = self._single_faiss_loader(os.path.join(db_folder, "text_db"))
                table_db_result = self._single_faiss_loader(os.path.join(db_folder, "table_db"))

                if not text_db_result["success"]:
                    raise ValueError(f"Текстовая база: {text_db_result['error']}")
                if not table_db_result["success"]:
                    raise ValueError(f"Табличная база: {table_db_result['error']}")

                result["text_db"] = text_db_result["db"]
                result["table_db"] = table_db_result["db"]

            result["success"] = True
            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    def _single_faiss_loader(self, db_folder: str) -> Dict[str, Any]:
        """
        Возвращает словарь с ключами:
        - success: bool - флаг успеха операции
        - db: Optional[FAISS] - объект базы (при успехе)
        - is_e5: bool - флаг использования E5 (при успехе)
        - error: str - сообщение об ошибке (при неудаче)
        """
        result = {
            "success": False,
            "db": None,
            "is_e5_model": False,
            "error": ""
        }

        try:
            # 1. Проверка существования папки
            if not os.path.isdir(db_folder): raise FileNotFoundError(f"Папка {db_folder} не существует")

            # 2. Загрузка метаданных
            code, metadata = self._load_metadata(db_folder)
            if metadata is None: raise ValueError("Невалидные метаданные базы")

            result["is_e5_model"] = metadata.get("is_e5_model", False)

            # 3. Проверка файлов FAISS
            required_files = ["index.faiss", "index.pkl"]
            missing = [f for f in required_files if not os.path.exists(os.path.join(db_folder, f))]
            if missing: raise FileNotFoundError(f"Отсутствуют файлы: {missing}")

            # 4. Загрузка эмбеддингов
            res, embeddings = self._load_embeddings(metadata)
            if embeddings is None: raise RuntimeError("Ошибка загрузки модели эмбеддингов")

            # 5. Основная загрузка
            result["db"] = FAISS.load_local(
                db_folder,
                embeddings,
                allow_dangerous_deserialization=True
            )

            result["success"] = True
            return result

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            return result

    def merge_databases(self, input_folders: List[str], output_folder: str) -> tuple:
        """
        Объединяет несколько FAISS-баз с проверкой совместимости
        Возвращает (success: bool, message: str)
        """
        try:
            # 1. Проверка минимального количества баз
            if len(input_folders) < 2: return False, "Необходимо минимум 2 базы для объединения"

            # 2. Загрузка и проверка метаданных
            result, main_meta = self._load_metadata(input_folders[0])
            if not main_meta: return False, result

            # 3. Проверка совместимости всех баз
            for folder in input_folders[1:]:
                result, current_meta = self._load_metadata(folder)
                if not self._check_compatibility(main_meta, current_meta):
                    msg = f"Несовместимые базы:\n{main_meta['embedding_model']}\nи\n{current_meta['embedding_model']}"
                    return False, msg

            # 4. Загрузка эмбеддингов
            err_code, embeddings = self._load_embeddings(main_meta)
            if not embeddings: return False, err_code

            # 5. Объединение баз
            merged_db = self._merge_faiss_indexes(input_folders, embeddings)

            # 6. Сохранение результата
            merged_db.save_local(output_folder)
            self._save_merged_metadata(output_folder, main_meta)

            return True, f"Базы успешно объединены в {output_folder}"

        except Exception as e:
            return False, f"Критическая ошибка: {str(e)}"

    @staticmethod
    def _load_metadata(folder: str) -> Any | None:
        """Загружает метаданные из папки с базой"""
        meta_path = os.path.join(folder, "metadata.json")
        try:
            with open(meta_path, "r") as f:
                return "Успешно", json.load(f)
        except FileNotFoundError:
            return f"Файл метаданных не найден: {meta_path}", None
        except json.JSONDecodeError as e:
            return f"Ошибка формата JSON: {e}", None
        except PermissionError:
            return f"Нет прав на чтение файла: {meta_path}", None
        except Exception as e:
            return f"Неизвестная ошибка: {e}", None

    def metadata_loader(self, folder):
        return self._load_metadata(folder)

    @staticmethod
    def _check_compatibility(meta1: dict, meta2: dict) -> bool:
        """Проверяет совместимость метаданных двух баз"""
        required_keys = [
            'embedding_model',
            'model_type',
            'normalized',
            'distance_strategy',
            'is_e5_model'
        ]

        for key in required_keys:
            if meta1.get(key) != meta2.get(key): return False
        return True

    def _load_embeddings(self, metadata: dict) -> (tuple[str, None] | tuple[str, HuggingFaceEmbeddings] |
                                                   tuple[str, OpenAIEmbeddings]):
        """Инициализирует модель эмбеддингов на основе метаданных"""
        try:
            model_type = metadata['model_type']
            model_name = metadata['embedding_model']
            if metadata['model_type'] == "openai":
                return "Успешно", OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key,
                    base_url=self.api_url
                )

            elif metadata['model_type'] == "huggingface":
                return "Успешно", HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={'normalize_embeddings': metadata['normalized']}
                )
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Доступные варианты: openai, huggingface")


        except ValueError as e: # Специфичная обработка ошибок валидации
            return f"Ошибка валидации: {str(e)}", None
        except KeyError as e:
            return f"Отсутствует обязательное поле в метаданных: {str(e)}", None
        except ImportError as e:
            return f"Ошибка импорта модуля: {str(e)}", None
        except Exception as e:
            return f"Неизвестная ошибка при загрузке эмбеддингов: {str(e)}", None

    @staticmethod
    def _merge_faiss_indexes(folders: List[str], embeddings: Embeddings) -> FAISS:
        """Объединяет FAISS-индексы"""
        merged_db = FAISS.load_local(folders[0], embeddings, allow_dangerous_deserialization=True)

        for folder in folders[1:]:
            current_db = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
            merged_db.merge_from(current_db)

        return merged_db

    @staticmethod
    def _save_merged_metadata(output_folder: str, meta: dict):
        """Создает расширенные метаданные для объединенной базы"""
        merged_meta = {
            "embedding_model": meta["embedding_model"],
            "model_type": meta["model_type"],
            "dimension": meta["dimension"],
            "normalized": meta["normalized"],
            "distance_strategy": meta["distance_strategy"],
            "is_e5_model": meta["is_e5_model"]
        }

        with open(os.path.join(output_folder, "metadata.json"), "w") as f:
            json.dump(merged_meta, f, indent=2)

#===================================================================================================
class Tester(DBConstructor):
    def __init__(self):
        super().__init__()

    def db_tester(self, db_markdown_text: list, system: str, user: str, verbose=False):

        questionnaire = []

        for chunk in db_markdown_text:
            if verbose:
                print(user + chunk.page_content)
                print('---------------------------------------------------------------------')
            request = f"{user}\n{chunk.page_content}"
            self.answer = self.request_to_openai(system, request, 0.5)
            if verbose:
                print(f"Вопросы от модели:\n{self.answer}")
                print('---------------------------------------------------------------------')
            questionnaire.append(self.answer)
            time.sleep(2.8)
        test_results = ''.join(questionnaire)
        return test_results

    def quest_handler(self, quest_file: str, system: str, user: str):
        # Метод для обработки пула тестовых вопросов по базе
        # и составления сводки по недостающей информации
        with open(quest_file, 'r') as qf:
            pull_questions = qf.read()
        # Делаю из пула вопросов langchain документ, чтобы подать в request_openai
        pull_questions = LangDoc(page_content=pull_questions, metadata={'pull': 'questions'})
        # Получаю сводку
        request = f"{user}\n{pull_questions}"
        self.summary = self.request_to_openai(system, request, 0)

        return self.summary

