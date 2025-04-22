import os

from rag_processor import *
import textwrap

constructor = DBConstructor()


root_folder = "/home/home/Projects/Uraltest"

folders = [os.path.join(root_folder, name) for name in os.listdir(root_folder)]

for folder in folders:
    file_names = [os.path.join(folder, name) for name in os.listdir(folder)]
    for name in file_names:
        out_path = f"{os.getcwd()}/FAISS/{name.split('.')[-2]}"
        chunk_file = f"{os.getcwd()}/Chunks/{name.split('.')[-2]}.txt"

        print(f"Документ: {name}")
        print(f"База: {out_path}")
        print(f"Чанки: {chunk_file}")
        print()

        parsed_chunks = constructor.document_parser(name)
        prepared_chunks = constructor.prepare_chunks(parsed_chunks)

        with open(chunk_file, "w") as file:
            for chunk in prepared_chunks:
                print(f"{chunk}\nНарушены связи: {constructor.validate_chunks(prepared_chunks)}"
                      f"\n------------------------------", file=file)

        success, msg = constructor.hybrid_vectorizator(docs=prepared_chunks, db_folder=out_path)
        print(success, msg)