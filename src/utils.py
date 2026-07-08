import re
import hashlib

from pathlib import Path
from pypdf import PdfReader

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


ROOT_DIR = Path(__file__).parent.parent


class PDFLoader:
    def __init__(self, file: str | Path):
        self.file_path = str(file)

    def load(self) -> list[Document]:
        reader = PdfReader(self.file_path)
        pages = []

        ref_pattern = re.compile(
            r"^\s*(References|REFERENCES|Bibliography)\s*$", 
            re.MULTILINE
        )
        for page in reader.pages:
            text = page.extract_text().strip()

            # references removing
            match = ref_pattern.search(text)
            if match:
                text = text[:match.start()]

            if text:
                pages.append(Document(
                    page_content=text,
                    metadata={"source": self.file_path, "page": page.page_number}
                ))
            if match:
                break
            
        return pages


def load_docs(
    data_dir: str,
    collection_name: str,
    db_dir: str = "./db",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embeddings_model: str = "nomic-embed-text"
):
    """
    Load, split and save in vector database

    Args:
        data_dir (str): directory with documents
        collection_name (str): name of a collection in vector database
        db_dir (str): directory of a vector database
        chunk_size (int): size of chunk for split 
        chunk_overlap (int): overlap between chunks
        embeddings_model (str): name of an embedding model
    """
    # read files
    data_dir = str(ROOT_DIR / data_dir)
    docs = [PDFLoader(file).load() for file in Path(data_dir).iterdir()]

    # split by chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_documents(doc))

    # set ids based on chunks (otherwise same chunks will 
    # appear several times in database with each script rerun)
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        content_hash = hashlib.md5((source + chunk.page_content).encode("utf-8")).hexdigest()
        ids.append(content_hash)

    # save to db
    embeddings = OllamaEmbeddings(model=embeddings_model)
    Chroma.from_documents(
        documents=chunks,
        ids=ids,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(ROOT_DIR / db_dir)
    )
