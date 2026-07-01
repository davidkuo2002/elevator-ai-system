import streamlit as st
import os
import PyPDF2
import base64
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="電梯 AI 專家診斷系統", layout="centered")

# --- 背景圖片快取 ---
@st.cache_data(show_spinner=False)
def get_cached_background():
    assets_dir = "./assets"
    if not os.path.exists(assets_dir): return None
    bg_files = [f for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if bg_files:
        with open(os.path.join(assets_dir, bg_files[0]), "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

encoded_string = get_cached_background()
if encoded_string:
    st.markdown(f"""<style>.stApp {{background-image: url("data:image/jpeg;base64,{encoded_string}"); background-size: cover;}}</style>""", unsafe_allow_html=True)

# --- 知識庫核心：合併載入 (Manuals + History) ---
@st.cache_resource(show_spinner=False)
def load_expert_knowledge_base(system_name):
    all_docs = []
    # 1. 載入該系統的手冊 (規範)
    manual_dir = os.path.join("./manuals", system_name)
    # 2. 載入所有歷史經驗 (實戰)
    history_dir = "./history"
    
    def add_docs(directory):
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(os.path.join(directory, file))
                    all_docs.extend(loader.load())
                    
    add_docs(manual_dir)
    add_docs(history_dir)
    
    if not all_docs: return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    split_docs = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma.from_documents(split_docs, embeddings)

# --- 初始化狀態 ---
if 'page' not in st.session_state: st.session_state.page = 1

# --- 頁面流程 ---
if st.session_state.page == 1:
    st.title("🛠️ 電梯 AI 專家診斷系統")
    system_options = ["請選擇...", "系統 A (傳統繼電器型)", "系統 B (微電腦變頻型)", "系統 C (最新無機房型)"]
    st.session_state.control_system = st.selectbox("選擇控制系統:", system_options)
    if st.button("確認進入"):
        if st.session_state.control_system != "請選擇...":
            st.session_state.page = 2
            st.rerun()

elif st.session_state.page == 2:
    st.title("📋 現場狀況回報")
    st.session_state.fault_code = st.text_input("故障碼:")
    st.session_state.fault_desc = st.text_area("現場狀況描述:")
    st.session_state.photo_value = st.camera_input("拍攝現場照片 (選用)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("回上頁"): st.session_state.page = 1; st.rerun()
    with col2:
        if st.button("開始分析"): st.session_state.page = 3; st.rerun()

elif st.session_state.page == 3:
    st.title("📊 診斷分析結果")
    with st.spinner("AI 正在整合原廠規範與歷史維修經驗..."):
        try:
            db = load_expert_knowledge_base(st.session_state.control_system)
            query = f"{st.session_state.fault_code} {st.session_state.fault_desc}"
            docs = db.similarity_search(query, k=4) if db else []
            context = "\n".join([d.page_content for d in docs])
            
            prompt = f"""你是一位資深電梯維修專家。
            請根據【原廠技術手冊規範】與【歷史維修經驗】綜合分析。
            若兩者有衝突，請優先遵循技術手冊規範，並補充前輩的實戰技巧。
            
            知識庫檢索內容：{context}
            現場狀況：{st.session_state.control_system}, 故障碼:{st.session_state.fault_code}, 描述:{st.session_state.fault_desc}
            
            請列出：1. 故障原因分析 2. 維修處置建議 3. 安全警示。保持精簡。"""

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
            response = llm.invoke([HumanMessage(content=prompt)])
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"分析錯誤: {e}")
            
    if st.button("結束並重置"):
        st.session_state.page = 1
        st.rerun()
