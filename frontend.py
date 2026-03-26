import streamlit as st
import os
import re

# ---------------- LANGCHAIN ----------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------------- YOUTUBE ----------------
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    VideoUnavailable,
    NoTranscriptFound
)

# ---------------- VECTOR DB ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ---------------- PAGE ----------------
st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("🎥 YouTube RAG Chatbot")


# ---------------- API KEY ----------------
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    st.error("❌ Add GROQ_API_KEY in .streamlit/secrets.toml")
    st.stop()


# ---------------- SESSION ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- EXTRACT VIDEO ID ----------------
def extract_video_id(url_or_id):
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, url_or_id)
        return match.group(1) if match else None
    return url_or_id


# ---------------- INPUT ----------------
video_input = st.text_input("Enter YouTube URL or Video ID")

video_id = extract_video_id(video_input)

if not video_id:
    st.info("👉 Enter valid YouTube URL or ID")
    st.stop()


# ---------------- SHOW VIDEO ----------------
st.video(f"https://www.youtube.com/watch?v={video_id}")


# ---------------- CACHE VECTOR DB ----------------
@st.cache_resource
def create_vector_store(video_id):
    try:
        # Try Hindi + English
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["hi", "en"]
        )
    except NoTranscriptFound:
        try:
            transcript_list = YouTubeTranscriptApi().fetch(video_id)
        except Exception:
            return None

    docs = []
    for chunk in transcript_list:
        docs.append({
            "text": chunk.text,
            "timestamp": int(chunk.start)
        })

    # ✅ FIXED CHUNKING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    split_docs = []
    for d in docs:
        chunks = splitter.split_text(d["text"])
        for c in chunks:
            split_docs.append({
                "page_content": c,
                "metadata": {"timestamp": d["timestamp"]}
            })

    # ✅ MULTILINGUAL EMBEDDING
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return FAISS.from_texts(
        [d["page_content"] for d in split_docs],
        embedding,
        metadatas=[d["metadata"] for d in split_docs]
    )


# ---------------- LOAD DB ----------------
vector_store = create_vector_store(video_id)

if vector_store is None:
    st.error("❌ Transcript not available for this video")
    st.stop()


# ✅ BETTER RETRIEVAL
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)


# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    streaming=True
)


# ---------------- PROMPT ----------------
prompt = PromptTemplate(
    template=(
        "You are a smart AI assistant that explains concepts from YouTube transcripts.\n\n"

        "GOAL:\n"
        "- Give a clear, direct, and helpful answer.\n"
        "- Use transcript as PRIMARY source.\n"
        "- If full answer is not present, infer logically from available context.\n\n"

        "RULES:\n"
        "- Do NOT repeat phrases like 'transcript does not provide'.\n"
        "- Do NOT loop or repeat same sentences.\n"
        "- Do NOT hallucinate unrelated info.\n"
        "- Be confident and natural like ChatGPT.\n"
        "- Answer in same language as question.\n\n"

        "FORMAT:\n"
        "- Start with clear explanation\n"
        "- Use bullet points if needed\n"
        "- Add timestamps ONLY if useful\n\n"

        "TRANSCRIPT:\n{content}\n\n"
        "CHAT HISTORY:\n{history}\n\n"
        "QUESTION:\n{question}\n\n"

        "FINAL ANSWER:"
    ),
    input_variables=["history", "content", "question"]
)


# ---------------- HELPERS ----------------
def clean_docs(docs):
    seen = set()
    result = []

    for d in docs:
        text = d.page_content.strip()
        if text not in seen:
            seen.add(text)
            ts = d.metadata.get("timestamp", 0)
            mins, secs = ts // 60, ts % 60
            result.append(f"[{mins}:{secs:02d}] {text}")

    return "\n\n".join(result)


def format_history(history):
    return "\n".join(
        [f"User: {h['question']}\nAI: {h['answer']}" for h in history]
    )


# ---------------- CHAIN ----------------
chain = (
    RunnableParallel({
        "content": RunnableLambda(lambda x: x["question"])
                   | retriever
                   | RunnableLambda(clean_docs),

        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: x["history"])
    })
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------- CLICKABLE TIMESTAMP ----------------
def make_clickable(text):
    def replace(match):
        mins, secs = match.group(1), match.group(2)
        total = int(mins) * 60 + int(secs)
        return f"[▶ {mins}:{secs}](https://www.youtube.com/watch?v={video_id}&t={total}s)"
    
    return re.sub(r"\[(\d+):(\d{2})\]", replace, text)


# ---------------- CHAT DISPLAY ----------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(make_clickable(chat["answer"]))


# ---------------- CHAT INPUT ----------------
user_query = st.chat_input("Ask something about the video...")

if user_query:

    with st.chat_message("user"):
        st.write(user_query)

    history_text = format_history(st.session_state.chat_history)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in chain.stream({
                "question": user_query,
                "history": history_text
            }):
                full_response += chunk
                placeholder.markdown(make_clickable(full_response) + "▌")

            placeholder.markdown(make_clickable(full_response))

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.markdown(full_response)

    # Save memory
    st.session_state.chat_history.append({
        "question": user_query,
        "answer": full_response
    })