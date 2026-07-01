import streamlit as st
import time
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# --- 頁面配置 ---
st.set_page_config(page_title="電梯 AI 診斷系統", layout="wide")
st.title("Elevator AI 診斷系統")

# --- 知識庫載入功能 (專為免費金鑰設計) ---
def load_knowledge_base():
    manuals_dir = "./manuals"
    documents = []
    
    # 確保資料夾存在，若無則自動建立防呆
    if not os.path.exists(manuals_dir):
        os.makedirs(manuals_dir)
        st.error("已自動建立 manuals 資料夾，但裡面沒有手冊。請至 GitHub 放入 PDF 後再試。")
        return None

    # 讀取資料夾內所有 PDF
    files = [f for f in os.listdir(manuals_dir) if f.endswith('.pdf')]
    if not files:
        st.error("找不到手冊檔案，請檢查 manuals 資料夾內是否有 PDF 檔。")
        return None

    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        try:
            progress_text.text(f"正在讀取 ({i+1}/{len(files)}): {file} ...")
            loader = PyPDFLoader(os.path.join(manuals_dir, file))
            documents.extend(loader.load())
            
            # 關鍵：免費金鑰必須放慢速度，避免觸發 429 錯誤
            time.sleep(1.5) 
            progress_bar.progress((i + 1) / len(files))
        except Exception as e:
            st.warning(f"無法讀取檔案 {file}: {e}")
            
    progress_text.text("正在建立向量特徵庫，請稍候...")
    
    try:
        # 使用最新支援的 text-embedding-004 模型進行向量化
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=st.secrets["GEMINI_API_KEY"]
        )
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"建立知識庫時發生錯誤 (可能金鑰無效或額度超限): {e}")
        return None

# --- 主程式邏輯 ---
if 'db' not in st.session_state:
    st.info("系統知識庫尚未載入。為了避免超用免費額度，請手動點擊下方按鈕載入。")
    if st.button("載入知識庫 (只需執行一次)"):
        with st.spinner("正在安全地轉換手冊為 AI 資料... (這需要一點時間)"):
            db = load_knowledge_base()
            if db is not None:
                st.session_state.db = db
                st.success("知識庫載入成功！")
                time.sleep(1) # 暫停一下讓畫面顯示成功訊息
                st.rerun()    # 重新整理頁面以顯示上傳功能
else:
    st.success("✅ 系統運作中：知識庫已就緒。")
    
    st.divider()
    st.subheader("🛠️ 故障維修報告單分析")
    
    # --- 您的核心功能區塊 ---
    uploaded_file = st.file_uploader("請上傳現場故障維修報告單 (PDF)", type="pdf")
    
    if uploaded_file:
        st.write(f"📄 已讀取檔案: **{uploaded_file.name}**")
        
        # 按鈕區塊
        if st.button("🚀 執行診斷分析", type="primary"):
            with st.spinner("AI 正在比對現場狀況與手冊內容..."):
                
                # ---------------------------------------------------
                # 這裡保留給您串接後續的 Gemini 分析邏輯
                # 可以從 st.session_state.db 進行檢索 (Similarity Search)
                # ---------------------------------------------------
                
                time.sleep(1) # 模擬運算時間，串接後可刪除此行
                st.write("分析完成！(此處為預留的分析結果顯示區)")
