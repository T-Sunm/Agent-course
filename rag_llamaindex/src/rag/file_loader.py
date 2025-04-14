from llama_index.readers.file import PDFReader
from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from llama_index.core import SimpleDirectoryReader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_file: str) -> List:
  """
  Load một file PDF sử dụng PDFReader của llama_index.

  :param pdf_file: Đường dẫn file PDF
  :return: Danh sách document được load từ file PDF.
  """
  try:
    loader = PDFReader()
    documents = loader.load_data(pdf_file)
  except Exception as e:
    print(f"Lỗi khi load {pdf_file}: {e}")
    documents = []
  return documents

def load_multiple_pdfs(pdf_files: List[str], workers: int = 4) -> List:
  """
  Sử dụng multiprocessing để load song song nhiều file PDF.

  :param pdf_files: Danh sách đường dẫn file PDF.
  :param workers: Số lượng worker/processes sử dụng.
  :return: Danh sách các document được load.
  """
  num_processes = min(multiprocessing.cpu_count(), workers)
  docs_loaded = []
  total_files = len(pdf_files)

  with multiprocessing.Pool(processes=num_processes) as pool:
    with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
      for result in pool.imap_unordered(load_pdf, pdf_files):
        docs_loaded.extend(result)
        pbar.update(1)
  return docs_loaded


if __name__ == "__main__":
    # Danh sách file PDF cần load (đảm bảo đường dẫn file chính xác)
  pdf_files = [
      "../../data_source/Attention Is All You Need.pdf",
      "../../data_source/BERT - Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf",
  ]
  documents = load_multiple_pdfs(pdf_files, workers=4)
  print(f"Đã load {len(documents)} document từ {len(pdf_files)} file PDF.")
