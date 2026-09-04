# import os

# from dotenv import load_dotenv
# from langchain_unstructured import UnstructuredLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_text_splitters import CharacterTextSplitter

# load_dotenv()


# if __name__ == "__main__":
#     print("Ingesting...")
#     loader = UnstructuredLoader("~/Desktop/langchain-course/mediumblog1.txt,chucking_strategy='basic'",max_characters=100000)
#     docment = loader.load()

#     print(f"Splilitting...")
#     text_spilter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
#     text=text_spilter.documents(docment)
#     print(f"creatded {len(text)} chunks")

import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")

    loader = UnstructuredLoader(
        os.path.expanduser("~/Desktop/langchain-course/mediumblog1.txt"),
        chunking_strategy="basic",
        max_characters=100000
    )

    document = loader.load()

    print("Splitting...")

    text_splitter = CharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0
    )

    text = text_splitter.split_documents(document)

    # print(f"Created {len(text)} chunks")
    # for i, chunk in enumerate(text):
    #     print(f"\n--- CHUNK {i+1} ---")
    #     print(chunk.page_content)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    print(".......")
    vectorstore = PineconeVectorStore.from_documents(
    text,
    embeddings,
    index_name=os.environ["INDEX_NAME"],
)
     
    print("finished....")