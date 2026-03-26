# -------------------- IMPORTS --------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# -------------------- STEP 1: GET TRANSCRIPT --------------------
video_id = "JxgmHe2NyeY"  # only ID

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("Transcript disabled")
    transcript = ""


# -------------------- STEP 2: SPLIT TEXT --------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([transcript])


# -------------------- STEP 3: EMBEDDINGS --------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------- STEP 4: VECTOR STORE --------------------
vector_store = FAISS.from_documents(chunks, embedding)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)


# -------------------- STEP 5: LLM --------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# -------------------- STEP 6: PROMPT --------------------
prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Answer ONLY from the transcript.\n"
        "If not found, say 'Content is not available'.\n\n"
        "Transcript:\n{content}\n\n"
        "Question:\n{question}\n"
        "Answer:"
    ),
    input_variables=["content", "question"]
)


# -------------------- STEP 7: FORMAT DOCS --------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------- STEP 8: CHAIN --------------------
parallel_chain = RunnableParallel({
    "content": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser


# -------------------- RUN --------------------
query = input("Ask your question: ")
response = main_chain.invoke(query)

print("\nAnswer:\n", response)