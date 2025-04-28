import os

from rag_processor import *

constructor = DBConstructor()

root = "/home/home/Projects/RAGProcessor/FAISS/Политики_по_вопросам_организации_деятельности"

folders_2_merge = os.listdir(root)

input_dirs = [os.path.join(root, each) for each in folders_2_merge]
out_dir = "/home/home/Projects/RAGProcessor/DB_FAISS/Политики_по_вопросам_организации_деятельности"

code, result = constructor.safe_hybrid_merge(input_dirs, out_dir)

print(result)