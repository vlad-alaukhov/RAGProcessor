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
        "pymupdf==1.23.7"
        "langchain-core==0.1.42"
        "langchain-community==0.0.24"
        "langchain-openai>=0.0.1"
        "tiktoken>=0.5.0"
        "faiss-cpu==1.8.0"
        "python-dotenv>=1.0.0"
        "requests>=2.26.0"
        "huggingface_hub==0.22.2"
        "sentence-transformers==2.7.0"
        "transformers>=4.26.0"
        "pydantic>=2.0.0"
        "loguru>=0.7.0"
        "langchain-huggingface==0.0.4"
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