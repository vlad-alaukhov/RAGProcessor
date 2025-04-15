from rag_processor import *
import textwrap

constructor = DBConstructor()

file_path = """/home/home/Projects/Uraltest/15 файлов для тестирования ПО СМК/Для GPT (копия).docx"""

parsed_chunks = constructor.document_parser(file_path)

prepared_chunks = constructor.prepare_chunks(parsed_chunks)

with open("Для_GPT_(копия).txt", "w") as file:
    for chunk in prepared_chunks:
        print(f"{chunk}\n------------------------------", file=file)