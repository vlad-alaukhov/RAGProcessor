import json
from abc import ABC
import fitz
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter     # рекурсивное разделение текста
from langchain.docstore.document import Document
# from langchain.docstore import Docstore
# from typing import Dict, Any
import tiktoken
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re                 # работа с регулярными выражениями
import requests
from dotenv import load_dotenv
import time
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Any, Dict
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

    def request_to_openai(self, system: str, request: str, temper: float):
        attempts = 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request}
            ],
            "temperature": temper,
        }

        while attempts < 3:
            try:
                time.sleep(10)
                # Отправляем POST-запрос к модели
                response = requests.post("https://api.vsegpt.ru:6070/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  # Проверка на наличие ошибок в запросе

                # Получаем ответ от модели
                response_data = response.json()
                result_text = response_data['choices'][0]['message']['content']
                return True, result_text

            except Exception as e:
                print(e)
                attempts += 1
                time.sleep(10)
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

    def split_recursive_from_markdown(self, documents_to_split: list, chunk_size: int) -> list:
        """ Делит список Langchain документов documents_to_split на чанки размером chunk_size
        методом RecursiveCharacterTextSplitter. Для этого вызывается метод split_text_recursive,
        реализованный в этом же классе.
        :param documents_to_split: Текст базы
        :param chunk_size: Размер чанка
        :return: список Langchain документов
        """
        source_chunks = []

        for each_document in documents_to_split:
            new_chunks = self.split_text_recursive(each_document.page_content, chunk_size)
            for each_element in new_chunks:
                for key in each_document.metadata:
                    each_element += each_document.metadata[key]
                source_chunks.append(Document(page_content=each_element, metadata=each_document.metadata))

        return source_chunks

    @staticmethod
    def split_markdown(db_text: str):
    # MarkDownHeader разметка базы из файла с дублированием заголовка в текст чанка

        hd_level = len(max(re.findall(r'#+', db_text)))
        print(hd_level)

        headers_to_split_on = [(f"{'#' * n}", f"H{n}") for n in range(1, hd_level+1 if hd_level >= 1 else 1)]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = markdown_splitter.split_text(db_text)

        for fragment in fragments:
            header = ''
            for key in fragment.metadata:
                header += f"{fragment.metadata[key]}. "
            fragment.page_content = header + fragment.page_content
            print(fragment)

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

    # x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
    # V
    def vectorizator_openai(self, docs: list, db_folder: str, model_name: str):
        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key = self.api_key,
            base_url = self.api_url)
        # embeddings.openai_api_key = self.api_key
        self.db = FAISS.from_documents(docs, embeddings)
        if self.db:
            self.db.save_local(db_folder)
            if os.path.exists(db_folder):
                return True, f"Векторизация прошла успешно в\n{db_folder}"
            else:
                return False, "Векторизация прошла, но загрузка не удалась"
        else:
            return False, "Векторизация не прошла"

    def vectorizator_sota(self, docs: list, db_folder: str, model_name: str):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        self.db = FAISS.from_documents(docs, embeddings)
        if self.db:
            self.db.save_local(db_folder)
            if os.path.exists(db_folder): return True, f"Векторизация прошла успешно в\n{db_folder}"
            else: return False, "Векторизация прошла, но загрузка не удалась"
        else:
            return False, "Векторизация не прошла"
    # ^
    # x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

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

    def faiss_loader(self, db_folder: str) -> Dict[str, Any]:
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
            metadata = self._load_metadata(db_folder)
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

    # x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
    # V
    def db_loader_from_sota(self, db_folder: str, model_name: str):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        self.db = FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)
        return self.db

    def db_loader_from_openai(self, db_folder):
        embs = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=self.api_key,
            base_url=self.api_url)
        self.db = FAISS.load_local(
            folder_path=db_folder,
            embeddings=embs,
            allow_dangerous_deserialization=True
        )
        return self.db
    # ^
    # x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

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
                current_meta = self._load_metadata(folder)
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

    @staticmethod
    def _check_compatibility(meta1: dict, meta2: dict) -> bool:
        """Проверяет совместимость метаданных двух баз"""
        required_keys = [
            'embedding_model',
            'model_type',
            'normalized',
            'distance_strategy'
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
            "distance_strategy": meta["distance_strategy"]
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
        pull_questions = Document(page_content=pull_questions, metadata={'pull': 'questions'})
        # Получаю сводку
        request = f"{user}\n{pull_questions}"
        self.summary = self.request_to_openai(system, request, 0)

        return self.summary

