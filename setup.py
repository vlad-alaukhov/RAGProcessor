from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag_processor",
    version="0.1.0",
    author="Vlad Alaukhov",
    author_email="your.email@example.com",
    description="RAG Processing Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlad-alaukhov/rag-processor",
    packages=find_packages(where="src"),
    py_modules=['rag_processor'],
    install_requires=[
        "pymupdf>=1.23.0",
        "pdfminer.six==20221105",
        "camelot-py==0.10.1",
        "langchain-core>=0.1.0",
        "langchain-community>=0.0.1",
        "langchain-openai>=0.0.1",
        "tiktoken>=0.5.0",
        "faiss-cpu>=1.7.0",
        "python-dotenv>=1.0.0",
        "requests>=2.26.0",
        "sentence-transformers>=2.2.0",
        "python-docx>=1.1.2",
        "pandas>=2.2.3",
        "openpyxl",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    keywords="rag ai nlp document-processing",
    project_urls={
        "Documentation": "https://github.com/vlad_alaukhov/rag-processor",
        "Source Code": "https://github.com/vlad_alaukhov/rag-processor"
    },
)