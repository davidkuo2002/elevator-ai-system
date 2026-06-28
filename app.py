import streamlit as st
import os
import time
import random
from google import genai
# 強制將保險箱抓到的金鑰傳入 api_key 參數中
client = genai.Client(api_key=API_KEY)
from PIL import Image

# 載入 PDF 處理與本機資料庫的基礎工具
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ==========================================
# 🔐 安全設定區 (使用官方標準讀取機制)
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("⚠️ 系統偵測不到 API 金鑰！請確保已在 Streamlit 後台的 Secrets 中設定 `GEMINI_API_KEY`。")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUALS_DIR = os.path.join(BASE_DIR, "manuals")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")  
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(MANUALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# ==========================================

# 初始化官方標準 Google GenAI 客戶端
client = genai.Client(api_key=API_KEY)
os.environ["GEMINI_API_KEY"] = API_KEY

# ------------------------------------------
# 特徵轉換包裝類別
# ------------------------------------------
class GenAIEmbeddingsWrapper:
    def _embed_with_retry(self, model: str, contents: list[str]) -> any:
        max_retries = 6      
        base_delay = 2       
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=contents
                )
                return response
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 20  
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = self._embed_with_retry(model="gemini-embedding-001", contents=batch_texts)
            all_embeddings.extend([embedding.values for embedding in response.embeddings])
            time.sleep(0.5)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self._embed_with_retry(model="gemini-embedding-001", contents=[text])
        return response.embeddings[0].values

@st.cache_resource
def setup_rag_database_multi_file():
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        try:
            embeddings = GenAIEmbeddingsWrapper()
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            return vectorstore, None
        except Exception as e:
            return None, f"讀取本機已快取知識庫時發生錯誤：{e}"

    manual_files = [f for f in os.listdir(MANUALS_DIR) if f.lower().endswith(".pdf")]
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(".pdf")]
    
    if not manual_files and not report_files:
        return None, "知識庫資料夾內空空如也！請至少在 `manuals` 放入一份官方手冊 PDF。"
    
    try:
        all_documents = []
        for filename in manual_files:
            loader = PyPDFLoader(os.path.join(MANUALS_DIR, filename))
            all_documents.extend(loader.load())
            
        for filename in report_files:
            loader = PyPDFLoader(os.path.join(REPORTS_DIR, filename))
            all_documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        splits = text_splitter.split_documents(all_documents)
        
        embeddings = GenAIEmbeddingsWrapper()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
        return vectorstore, None
    except Exception as e:
        return None, f"後台建立知識庫時發生錯誤：{e}"

# ------------------------------------------
# 前端網頁介面 
# ------------------------------------------
st.set_page_config(page_title="電梯維修 AI 智聯專家系統", page_icon="🛠️", layout="wide")

st.sidebar.markdown("### 📚 知識庫整合狀態")
with st.sidebar.spinner('正在檢查並喚醒官方手冊知識庫中...'):
    db, error_message = setup_rag_database_multi_file()

if error_message:
    st.sidebar.error(error_message)
else:
    st.sidebar.success("✅ 官方手冊知識庫已就緒！")

st.title("🛠️ 電梯維修 AI 智聯專家系統 (官方手冊標準版)")
st.write("本系統已整合「官方技術手冊」。請提供現場線索，AI 將自動對齊原廠規範進行診斷。")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🔍 現場線索輸入")
    uploaded_file = st.file_uploader("📸 拍照上傳現場照片 (支援 jpg, png)", type=["jpg", "jpeg", "png"])
    audio_file = st.audio_input("🎙️ 語音口述現場狀況：")
    user_text = st.text_area("⌨️ 現場狀況描述 (打字區)：", height=100)

system_instruction = """
你是一位嚴謹的電梯維修專家。請根據下方提供的「參考手冊」，檢視現場工程師提供的文字、語音或照片：
1. 找出並核對資料中對應的錯誤代碼、組件名稱或故障說明。
2. 條列出符合官方規範的排查步驟與現場維修安全守則。
- 如果參考資料中完全沒有提及此狀況，請誠實告知「手冊未記載此項資訊」，切勿自行瞎掰。
"""

with col2:
    st.markdown("### 📋 AI 專家檢修報告")
    if st.button("🚀 開始對齊與診斷分析", use_container_width=True):
        if not (uploaded_file or audio_file or user_text.strip()):
            st.warning("⚠️ 請至少提供一項現場資訊！")
        elif db is None:
            st.error("知識庫未成功建立，請檢查左側狀態。")
        else:
            with st.spinner('正在調閱手冊原文，比對分析中...'):
                try:
                    actual_problem = ""
                    if user_text.strip():
                        actual_problem += f"【工程師文字描述】：{user_text}\n"
                    
                    if audio_file:
                        st.toast("正在轉換現場語音...", icon="🎙️")
                        audio_bytes = audio_file.read()
                        from google.genai import types
                        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                        transcript_response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[audio_part, "請精準辨識這段語音，直接輸出繁體中文逐字稿，不需任何額外說明。"]
                        )
                        spoken_text = transcript_response.text
                        st.info(f"🗣️ 語音辨識結果：{spoken_text}")
                        actual_problem += f"【工程師語音描述】：{spoken_text}\n"

                    search_query = actual_problem if actual_problem.strip() else "請根據照片中的異常現象，提供可能的故障原因。"
                    relevant_docs = db.as_retriever(search_kwargs={"k": 4}).invoke(search_query)
                    
                    context_str = "\n\n--- 參考手冊片段 ---\n"
                    for i, doc in enumerate(relevant_docs):
                        context_str += f"[參考文本段落 {i+1}]:\n{doc.page_content}\n"
                    context_str += "----------------------\n"
                    
                    full_content = [system_instruction]
                    if actual_problem.strip():
                        full_content.append(actual_problem)
                    full_content.append(context_str)
                    
                    if uploaded_file:
                        img = Image.open(uploaded_file)
                        st.image(img, caption="上傳的現場照片", use_container_width=True)
                        full_content.append(img)
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=full_content
                    )
                    
                    st.success("分析完成！")
                    st.write(response.text)
                    
                    with st.expander("🔍 檢視本次 AI 參考的知識庫原文片段"):
                        st.write(context_str)
                        
                except Exception as e:
                    st.error(f"分析時發生錯誤：{e}")
