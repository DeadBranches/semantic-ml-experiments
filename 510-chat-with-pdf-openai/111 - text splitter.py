from langchain.text_splitter import RecursiveCharacterTextSplitter
from a_mupdf_loader import pdf_loader

documents = pdf_loader(
    filepath="../pdfs/2303.17760.pdf"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
texts = text_splitter.create_documents(documents)

index = 0
while index < 10:
    index += 1
    # convert document object into JSON
    #

    print(f"{texts[index]}\n\n")
