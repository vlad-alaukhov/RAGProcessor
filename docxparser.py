import os

from rag_processor import *
import textwrap

constructor = DBConstructor()


root_folder = "/home/home/Projects/Uraltest"

folders = [os.path.join(root_folder, name) for name in os.listdir(root_folder)]

for folder in folders:
    file_names = [os.path.join(folder, name) for name in os.listdir(folder)]
    for name in file_names:
        file_name = '/'.join(name.split('.')[-2].split('/')[5:])
        out_path = f"{os.getcwd()}/FAISS/{file_name}"
        chunk_file = f"{os.getcwd()}/Chunks/{file_name}.txt"

        os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

        print(f"Документ: {name}")
        print(f"База: {out_path}")
        print(f"Чанки: {chunk_file}")
        print()

        parsed_chunks = constructor.document_parser(name)
        prepared_chunks = constructor.prepare_chunks(parsed_chunks, name)

        with open(chunk_file, "a") as file:
            file.writelines(f"{prepared_chunks}\nНарушены связи: {constructor.validate_chunks(prepared_chunks)}"
                      f"\n------------------------------")


        success, msg = constructor.hybrid_vectorizator(docs=prepared_chunks, db_folder=out_path)
        print(success, msg)