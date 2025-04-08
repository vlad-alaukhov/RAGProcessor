from rag_processor import *
import textwrap

constructor = DBConstructor()

file_path = """/home/home/Projects/Uraltest/15 файлов для тестирования ПО СМК/Для GPT (копия).docx"""

result = constructor.document_parser(file_path)
print(textwrap.fill(result, 150))
