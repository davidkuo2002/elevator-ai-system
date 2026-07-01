import streamlit as st
import os
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

# --- 知識庫核心 (修復版：保護中文編碼，精準切片) ---
@st.cache_resource(show_spinner=False)
def load_expert_knowledge_base(system_name):
    all_docs = []
    def add_docs(directory):
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(os.path.join(directory, file))
                    # 拔除 encode('utf-8', 'ignore')，保留原始 PDF 文字，防止關鍵字被吃掉
                    all_docs.extend(loader.load())
    
    add_docs(os.path.join("./manuals", system_name))
    add_docs("./history")
    
    if not all_docs: return None
    
    # ⚡ 核心修正：指定中文字元分隔符，絕對不切斷單一中文字的編碼
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""],
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma.from_documents(split_docs, embeddings)

# --- 初始化 ---
if 'page' not in st.session_state: st.session_state.page = 1
if 'board_code' not in st.session_state: st.session_state.board_code = ""
if 'inverter_code' not in st.session_state: st.session_state.inverter_code = ""
if 'fault_desc' not in st.session_state: st.session_state.fault_desc = ""

# --- 頁面邏輯 ---
if st.session_state.page == 1:
    st.title("🛠️ 電梯 AI 專家診斷系統")
    system_options = ["請選擇...", "系統 A (CHIMAX)", "系統 B (HPM)", "系統 C (IED)"]
    st.session_state.control_system = st.selectbox("選擇控制系統:", system_options)
    if st.button("確認進入"):
        if st.session_state.control_system != "請選擇...":
            st.session_state.page = 2
            st.rerun()

elif st.session_state.page == 2:
    st.title("📋 現場狀況回報")
    st.session_state.board_code = st.text_input("主機板故障碼:", value=st.session_state.board_code)
    st.session_state.inverter_code = st.text_input("變頻器故障碼:", value=st.session_state.inverter_code)
    st.info("💡 提示：請使用手機輸入法的「麥克風」圖示進行語音轉文字。")
    st.session_state.fault_desc = st.text_area("現場狀況描述:", value=st.session_state.fault_desc, height=150)
    st.session_state.uploaded_file = st.file_uploader("上傳現場照片", type=['jpg', 'jpeg', 'png'])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("回上頁"): st.session_state.page = 1; st.rerun()
    with col2:
        if st.button("開始分析"): st.session_state.page = 3; st.rerun()

elif st.session_state.page == 3:
    st.title("📊 診斷分析結果")
    with st.spinner("AI 正在深度比對手冊與經驗..."):
        try:
            db = load_expert_knowledge_base(st.session_state.control_system)
            
            # 優化搜尋語句，讓 Chroma 更好抓取關鍵字
            query = f"主機板故障碼:{st.session_state.board_code} 變頻器故障碼:{st.session_state.inverter_code} 狀況:{st.session_state.fault_desc}"
            
            # ⚡ 核心修正：退回最安全的 k=4，避免雜訊污染 AI 判斷
            docs = db.similarity_search(query, k=4) if db else []
            context = "\n".join([d.page_content for d in docs])
            
            # 簡化且明確的 Prompt，防止 AI 邏輯崩潰
            prompt = f"""你是一位資深電梯維修專家。請根據下方知識庫內容，提供精簡、準確的繁體中文維修建議。
            
            【任務規則】
            1. 優先尋找與「{st.session_state.board_code}」或「{st.session_state.inverter_code}」完全相符的官方定義。
            2. 若知識庫中找不到對應的故障碼定義，請誠實回答「手冊中未找到此故障碼」，切勿自行編造。
            3. 若文字中有出現零星錯字，請自行判斷上下文修正，嚴禁輸出亂碼。
            
            【知識庫檢索內容】
            {context}
            
            【現場狀況】
            系統:{st.session_state.control_system}
            主機板故障碼:{st.session_state.board_code}
            變頻器故障碼:{st.session_state.inverter_code}
            描述:{st.session_state.fault_desc}
            """

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
            
            if st.session_state.uploaded_file:
                image_data = base64.b64encode(st.session_state.uploaded_file.getvalue()).decode('utf-8')
                message = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ])
            else:
                message = HumanMessage(content=prompt)

            response = llm.invoke([message])
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"分析發生錯誤: {e}")
            
    if st.button("結束並重置"):
        st.session_state.page = 1
        st.session_state.uploaded_file = None
        st.session_state.board_code = ""
        st.session_state.inverter_code = ""
        st.session_state.fault_desc = ""
        st.rerun()
