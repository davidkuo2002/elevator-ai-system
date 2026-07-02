import streamlit as st
import os
import base64
import re  # 👈 新增：用於清理文本雜訊
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

# --- 知識庫核心 (還原流暢度與純淨文本) ---
@st.cache_resource(show_spinner=False)
def load_expert_knowledge_base(system_name):
    all_docs = []
    def add_docs(directory):
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if file.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    # 💡 清理 PDF 抽字時可能產生的控制字元與不明隱形雜訊
                    for page in pages:
                        page.page_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', page.page_content)
                    all_docs.extend(pages)
                elif file.lower().endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    all_docs.extend(loader.load())
                    
    add_docs(os.path.join("./manuals", system_name))
    add_docs("./history")
    
    if not all_docs: return None
    
    # 穩定的中文自然段落切片
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，"],
        chunk_size=700,
        chunk_overlap=120
    )
    split_docs = splitter.split_documents(all_docs)
    
    # ⚡ 升級為 Google 最新代 text-embedding-004 模型，語意辨識大幅提升
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
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
            
            query = f"主機板故障碼 {st.session_state.board_code} 變頻器故障碼 {st.session_state.inverter_code} {st.session_state.fault_desc}"
            docs = db.similarity_search(query, k=4) if db else []
            context = "\n".join([d.page_content for d in docs])
            
            # 💡 還原最初深受好評的「專業老前輩維修專家」Prompt 風格
            prompt = f"""你是一位擁有20年現場維修經驗的資深電梯技術專家。
            請詳細閱讀下方的技術手冊規範與過往故障處置單，針對現場人員回報的故障碼與狀況，給出最精準、最具實戰價值的繁體中文維修指導。
            
            手冊與經驗參考資料：
            {context}
            
            現狀回報：
            - 控制系統：{st.session_state.control_system}
            - 主機板故障碼：{st.session_state.board_code}
            - 變頻器故障碼：{st.session_state.inverter_code}
            - 現場狀況描述：{st.session_state.fault_desc}
            
            請條列式清晰輸出：
            1. 故障原因深度分析（指出可能損壞的零件、迴路或接點）
            2. 現場維修處置步驟（按檢查順序排列，寫出具體動作）
            3. 現場作業安全警示
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
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                st.warning("⏳ 系統目前分析請求較多（已達免費 API 流量上限）。請您稍等 1 分鐘後，再次點擊分析！")
            else:
                st.error(f"分析發生錯誤: {e}")
            
    if st.button("結束並重置"):
        st.session_state.page = 1
        st.session_state.uploaded_file = None
        st.session_state.board_code = ""
        st.session_state.inverter_code = ""
        st.session_state.fault_desc = ""
        st.rerun()
