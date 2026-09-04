import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatOllama(model="qwen3:1.7b", base_url="http://localhost:11434")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# IMPLEMENTATION 1: Without LCEL
# ================================================
#  without lcel :langchain expression language
def retreival_chain_without_lcel(query:str):
    """ 
    Simple retrieval chain wihtout LECL 
    Manually retrevies Docs,fromat them,and generate an response
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # retrieve relevant docs
    docs=retriever.invoke(query)
    # format docs into a context string
    context=format_docs(docs)
    # format the prompt and context and question
    message=prompt_template.format_messages(context=context,question=query)
    # invoke the llm
    response=llm.invoke(message)


# ================================================
# IMPLEMENTATION 2: Using LCEL
# ================================================
def retreival_chain_with_lcel(query:str):
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    # idhi same working from the kindha unna flow chat laga run avuthundhi
    retrieval_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        |prompt_template 
        | llm 
        | StrOutputParser()
    )
    return retrieval_chain
    
if __name__ == "__main__":
    print("retrieving..")

    query = "what is Pinecone in machine learning?"

# -----------
# option 0: raw invocation without rag
# -----------
    print("\n" + "=" * 70)
    print("printing raw llm invocation without rag")
    print("=" * 70)
    result_raw = llm.invoke(query)
    print("\nAnswer")
    print(result_raw.content)

# ----------
# option 1:using the impelmentation without LCEL
# ----------
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retreival_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

#----------
# option 2:using the impelmentation with LCEL
#----------
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)







#   print("\n" + "=" * 70)
#     print("IMPLEMENTATION 2: With LCEL - Better Approach")
#     print("=" * 70)
#     print("Why LCEL is better:")
#     print("- More concise and declarative")
#     print("- Built-in streaming: chain.stream()")
#     print("- Built-in async: chain.ainvoke()")
#     print("- Easy to compose with other chains")
#     print("- Better for production use")
#     print("=" * 70)


    #                 USER QUESTION
    #                       │
    #                       ▼
    #           {"question": "..."}
    #                       │
    #                       ▼
    #         RunnablePassthrough.assign
    #                       │
    #          ┌────────────┴────────────┐
    #          │                         │
    #          ▼                         │
    #  itemgetter("question")            │
    #          │                         │
    #          ▼                         │
    #      retriever                     │
    #          │                         │
    #          ▼                         │
    #   [Document, Document, ...]        │
    #          │                         │
    #          ▼                         │
    #     format_docs                    │
    #          │                         │
    #          ▼                         │
    #       context                     │
    #          │                         │
    #          └────────────┬────────────┘
    #                       ▼
    #          {
    #            question: "...",
    #            context: "..."
    #          }
    #                       │
    #                       ▼
    #               prompt_template
    #                       │
    #                       ▼
    #                      LLM
    #                       │
    #                       ▼
    #               StrOutputParser
    #                       │
    #                       ▼
    #                FINAL STRING