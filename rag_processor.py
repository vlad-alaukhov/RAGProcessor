from abc import ABC
import fitz
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter     # рекурсивное разделение текста
from langchain.docstore.document import Document
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re                 # работа с регулярными выражениями
import requests
from dotenv import load_dotenv
import time
from langchain_huggingface import HuggingFaceEmbeddings

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

    def vectorizator_openai(self, docs: list, db_folder: str, model_name: str):
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key = self.api_key,
            openai_api_base = self.api_url)
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
            openai_api_key=self.api_key,
            openai_api_base=self.api_url)
        self.db = FAISS.load_local(
            folder_path=db_folder,
            embeddings=embs,
            allow_dangerous_deserialization=True
        )
        return self.db

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

