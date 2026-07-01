import streamlit as st
import time
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# --- 頁面配置 ---
st.set_page_config(page_title="電梯 AI 診斷系統", layout="wide")
st.title("Elevator AI 診斷系統")

# --- 知識庫載入功能 (改用開源免費模型) ---
def load_knowledge_base():
    manuals_dir = "./manuals"
    documents = []
    
    if not os.path.exists(manuals_dir):
        os.makedirs(manuals_dir)
        st.error("已自動建立 manuals 資料夾，請至 GitHub 放入 PDF 後再試。")
        return None

    files = [f for f in os.listdir(manuals_dir) if f.endswith('.pdf')]
    if not files:
        st.error("找不到手冊檔案，請檢查 manuals 資料夾內是否有 PDF 檔。")
        return None

    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # 讀取 PDF
    for i, file in enumerate(files):
        try:
            progress_text.text(f"正在讀取 ({i+1}/{len(files)}): {file} ...")
            loader = PyPDFLoader(os.path.join(manuals_dir, file))
            documents.extend(loader.load())
            progress_bar.progress((i + 1) / len(files))
        except Exception as e:
            st.warning(f"無法讀取檔案 {file}: {e}")
            
    progress_text.text("正在使用開源模型建立特徵庫 (免金鑰、無限制)...")
    
    try:
        # 關鍵修改：改用本地開源多語言模型，完全繞過 Google API 的限制！
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"建立知識庫時發生錯誤: {e}")
        return None

# --- 主程式邏輯 ---
if 'db' not in st.session_state:
    st.info("系統知識庫尚未載入。")
    if st.button("載入知識庫 (本地免費運行)"):
        with st.spinner("正在轉換手冊為 AI 資料... (第一次執行會下載開源模型，請稍候)"):
            db = load_knowledge_base()
            if db is not None:
                st.session_state.db = db
                st.success("知識庫載入成功！不再受 Google 額度限制。")
                time.sleep(1.5) 
                st.rerun()    
else:
    st.success("✅ 系統運作中：知識庫已就緒。")
    
    st.divider()
    st.subheader("🛠️ 故障維修報告單分析")
    
    uploaded_file = st.file_uploader("請上傳現場故障維修報告單 (PDF)", type="pdf")
    
    if uploaded_file:
        st.write(f"📄 已讀取檔案: **{uploaded_file.name}**")
        
        if st.button("🚀 執行診斷分析", type="primary"):
            with st.spinner("AI 正在比對現場狀況與手冊內容..."):
                time.sleep(1) 
                st.write("分析完成！(此處為預留的分析結果顯示區)")
