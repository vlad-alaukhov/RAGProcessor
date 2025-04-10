from rag_processor import *
import textwrap

constructor = DBConstructor()

file_path = """/home/home/Projects/Uraltest/15 файлов для тестирования ПО СМК/Для GPT (копия).docx"""

result = constructor.document_parser(file_path)
with open("Для_GPT_(копия).txt", "w") as file:
    for each in result:
        # print(textwrap.fill(f"page_content=\"{each.page_content}\"\n\nmetadata={each.metadata}\n=======================", 150), file="Для_GPT_(копия).txt")
        print(f"page_content=\"{each.page_content}\"\nmetadata={each.metadata}\n------------------------------", file=file)
#print(f"metadata={each.metadata}")
