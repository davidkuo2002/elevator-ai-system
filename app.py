import streamlit as st
import time
import os
import PyPDF2
import base64
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- 頁面配置 ---
st.set_page_config(page_title="電梯 AI 診斷系統", layout="centered")

# --- 初始化「頁面狀態」與「暫存變數」 ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'control_system' not in st.session_state:
    st.session_state.control_system = ""
if 'fault_code' not in st.session_state:
    st.session_state.fault_code = ""
if 'fault_desc' not in st.session_state:
    st.session_state.fault_desc = ""
if 'photo_value' not in st.session_state:
    st.session_state.photo_value = None

# --- 側邊欄：背景與後台上傳區 ---
st.sidebar.title("⚙️ 系統後台設定")
bg_url = st.sidebar.text_input("輸入背景圖片網址 (URL) 即可更換", "")
if bg_url:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.sidebar.divider()
st.sidebar.subheader("📄 歷史維修報告擴充")
st.sidebar.write("上傳過去的維修報告單，AI 會將其作為額外參考。")
uploaded_report = st.sidebar.file_uploader("上傳維修報告 (PDF)", type="pdf")

# 解析上傳的維修報告單內容
report_text = ""
if uploaded_report:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_report)
        for page in pdf_reader.pages:
            report_text += page.extract_text() + "\n"
        st.sidebar.success("✅ 報告單內容已讀取！")
    except Exception as e:
        st.sidebar.error("讀取報告失敗。")

# --- 知識庫載入功能 (本地開源模型，免 API 費用) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    manuals_dir = "./manuals"
    documents = []
    
    if not os.path.exists(manuals_dir):
        os.makedirs(manuals_dir)
        return None
        
    files = [f for f in os.listdir(manuals_dir) if f.endswith('.pdf')]
    if not files:
        return None
        
    for file in files:
        loader = PyPDFLoader(os.path.join(manuals_dir, file))
        documents.extend(loader.load())
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma.from_documents(documents, embeddings)

st.session_state.db = load_knowledge_base()

# ==========================================
# 網頁流程路由
# ==========================================

# ----------------- 第一頁：選擇控制系統 -----------------
if st.session_state.page == 1:
    st.title("🛠️ 第一步：選擇控制系統")
    if st.session_state.db is None:
        st.warning("⚠️ 尚未偵測到原廠手冊，請確認 manuals 資料夾內有 PDF 檔案。")
        
    control_system = st.selectbox(
        "請確認目前維修的電梯廠牌與控制系統：",
        ["請選擇...", "系統 A (傳統繼電器型)", "系統 B (微電腦變頻型)", "系統 C (最新無機房型)"],
        index=0 if not st.session_state.control_system else ["請選擇...", "系統 A (傳統繼電器型)", "系統 B (微電腦變頻型)", "系統 C (最新無機房型)"].index(st.session_state.control_system)
    )
    
    if st.button("確認，進入下一步 ➡️", type="primary"):
        if control_system == "請選擇...":
            st.error("請先選擇一個控制系統！")
        else:
            st.session_state.control_system = control_system
            st.session_state.page = 2
            st.rerun()

# ----------------- 第二頁：輸入現場狀況 -----------------
elif st.session_state.page ==
