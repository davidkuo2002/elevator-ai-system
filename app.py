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

# ==========================================
# 🎨 系統背景設定 (加入快取機制，防止重複讀取)
# ==========================================
@st.cache_data(show_spinner=False)
def get_cached_background_base64(assets_dir):
    """
    此函式加入了快取機制。
    只有在系統初次啟動、或是 assets 資料夾內容有變更時，才會真正執行裡面的程式碼。
    平常切換網頁或重整時，Streamlit 會直接從記憶體回傳結果，效率極高。
    """
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        return None
    
    # 抓取資料夾內的圖片檔
    bg_files = [f for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if bg_files:
        bg_path = os.path.join(assets_dir, bg_files[0])
        try:
            with open(bg_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception:
            return None
    return None

def set_local_background():
    assets_folder = "./assets"
    # 呼叫帶有快取的函式
    encoded_string = get_cached_background_base64(assets_folder)
    
    if encoded_string:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# 執行背景設定
set_local_background()

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

# --- 側邊欄：後台擴充維修報告區 ---
st.sidebar.title("⚙️ 系統後台設定")
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

# --- 知識庫載入功能 (本地開源模型) ---
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
elif st.session_state.page == 2:
    st.title("📋 第二步：現場狀況蒐集")
    st.info(f"目標系統：**{st.session_state.control_system}**")
    
    st.session_state.fault_code = st.text_input("輸入主機版顯示之故障碼 (選填)", value=st.session_state.fault_code)
    
    st.write("請提供其他現場資訊（可單選或複選）：")
    use_text = st.checkbox("📝 文字描述現場狀況")
    use_voice = st.checkbox("🎤 語音輸入 (目前僅存檔，語音辨識未來擴充)")
    use_photo = st.checkbox("📷 拍攝/上傳異常燈號或現場照片")
    
    st.divider()
    
    if use_text:
        st.session_state.fault_desc = st.text_area("故障情況描述", value=st.session_state.fault_desc)
    else:
        st.session_state.fault_desc = ""
        
    if use_voice:
        audio_value = st.audio_input("錄製語音")
        
    if use_photo:
        st.session_state.photo_value = st.camera_input("拍攝現場照片")
        
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 回上一步"):
            st.session_state.page = 1
            st.rerun()
    with col2:
        if st.button("🚀 確認送出，開始分析", type="primary"):
            st.session_state.page = 3
            st.rerun()

# ----------------- 第三頁：AI 判斷分析結果 -----------------
elif st.session_state.page == 3:
    st.title("📊 第三步：AI 診斷分析結果")
    st.info(f"正在針對 **{st.session_state.control_system}** 進行多維度分析...")
    
    with st.spinner("AI 正在比對手冊、歷史報告與現場照片..."):
        try:
            search_query = f"系統:{st.session_state.control_system} 故障碼:{st.session_state.fault_code} 狀況:{st.session_state.fault_desc}"
            docs = st.session_state.db.similarity_search(search_query, k=3) if st.session_state.db else []
            manual_context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            你是一位資深的電梯維修工程師。請綜合以下資訊，給予現場維修人員最專業、安全的處置建議。
            
            【現場輸入資訊】
            - 控制系統：{st.session_state.control_system}
            - 故障碼：{st.session_state.fault_code}
            - 狀況描述：{st.session_state.fault_desc}
            
            【原廠技術手冊參考內容】
            {manual_context if manual_context else "無相關手冊資料。"}
            
            【歷史故障維修報告單參考】
            {report_text if report_text else "本次未提供歷史報告。"}
            
            請根據上述資料（若有提供現場照片，請一併分析），給出：
            1. 可能的故障原因。
            2. 建議的維修步驟與檢查點。
            3. 現場安全注意事項。
            """

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash", 
                google_api_key=st.secrets["GEMINI_API_KEY"]
            )
            
            if st.session_state.photo_value is not None:
                image_data = base64.b64encode(st.session_state.photo_value.getvalue()).decode('utf-8')
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                )
            else:
                message = HumanMessage(content=prompt)

            response = llm.invoke([message])
            
            st.success("✅ 分析完成！")
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"分析過程中發生錯誤：{e}")
            st.write("請確認您的 API Key 額度是否足夠，或檢查連線狀態。")
    
    st.divider()
    if st.button("🔄 結束並開啟新診斷", type="primary"):
        st.session_state.page = 1
        st.session_state.fault_code = ""
        st.session_state.fault_desc = ""
        st.session_state.photo_value = None
        st.rerun()
