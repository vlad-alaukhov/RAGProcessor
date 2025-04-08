from rag_processor import *
import textwrap

constructor = DBConstructor()

file_path = """/home/home/Projects/Uraltest/15 файлов для тестирования ПО СМК/Для GPT (копия).docx"""

result = constructor.document_parser(file_path)
for each in result:
    print(textwrap.fill(f"page_content=\"{each.page_content}\"", 150))
    print(f"metadata={each.metadata}")
