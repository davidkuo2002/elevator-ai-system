import streamlit as st
import os
import time
import random
import requests
import base64
from PIL import Image

# 載入 PDF 處理與本機資料庫的最基礎工具
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ==========================================
# 🔐 安全設定區 (使用原生 requests 繞過 Google 伺服器 Bug)
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

# ------------------------------------------
# 🚀 終極破解版：原生 HTTP 請求包裝類別 (完全捨棄官方 SDK)
# ------------------------------------------
class NativeRequestsEmbeddings:
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        max_retries = 6      
        base_delay = 2       
        # 強制將金鑰綁在網址參數，完全繞過標頭誤判 Bug
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        payload = {
            "requests": [
                {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload)
                if resp.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    st.sidebar.warning(f"⚠️ 觸發流量上限 (429)。系統將於 {delay:.1f} 秒後進行安全重試...")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return [item["values"] for item in data["embeddings"]]
            except Exception as e:
                if attempt == max_retries - 1:
                    error_detail = resp.text if 'resp' in locals() else str(e)
                    raise RuntimeError(f"API 請求失敗: {error_detail}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 20  
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self._embed_with_retry(batch_texts)
            all_embeddings.extend(embeddings)
            time.sleep(0.5)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_retry([text])[0]

@st.cache_resource
def setup_rag_database_multi_file():
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        try:
            embeddings = NativeRequestsEmbeddings()
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
        
        embeddings = NativeRequestsEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
        return vectorstore, None
    except Exception as e:
        return None, f"後台建立知識庫時發生錯誤：{e}"

# ------------------------------------------
# 前端網頁介面 
# ------------------------------------------
st.set_page_config(page_title="電梯維修 AI 智聯專家系統", page_icon="🛠️", layout="wide")

st.sidebar.markdown("### 📚 知識庫整合狀態")
with st.sidebar.spinner('正在檢查並喚醒雙軌知識庫中...'):
    db, error_message = setup_rag_database_multi_file()

if error_message:
    st.sidebar.error(error_message)
else:
    st.sidebar.success("✅ 官方手冊 + 歷史經驗知識庫已就緒！")

st.title("🛠️ 電梯維修 AI 智聯專家系統 (手冊與實戰經驗雙軌版)")
st.write("本系統已整合「官方技術手冊」與「前線維修報告」。請提供現場線索，AI 將自動對齊規範與過往實戰經驗進行診斷。")

# ==========================================
# 📥 知識庫經驗補給站
# ==========================================
with st.expander("📥 知識庫經驗補給站 (上傳過往維修報告、擴充 AI 大腦)", expanded=False):
    st.info("💡 **提示：** 當前線遇到特殊故障並排除後，請將結案報告(PDF)上傳至此。")
    uploaded_report = st.file_uploader("請選擇要匯入的維修紀錄、結案報告 (限 PDF)", type=["pdf"], key="report_uploader")
    
    if uploaded_report is not None:
        save_path = os.path.join(REPORTS_DIR, uploaded_report.name)
        if os.path.exists(save_path):
            st.warning(f"ℹ️ 檔案 `{uploaded_report.name}` 先前已匯入過。")
        else:
            if st.button("確認將此報告寫入本機知識庫", use_container_width=True):
                with st.spinner("正在解析報告並寫入..."):
                    try:
                        with open(save_path, "wb") as f:
                            f.write(uploaded_report.getbuffer())
                        loader = PyPDFLoader(save_path)
                        db.add_documents(RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50).split_documents(loader.load()))
                        st.success(f"🎉 成功！經驗已動態融入大腦。")
                        time.sleep(1.5)
                        st.rerun() 
                    except Exception as e:
                        st.error(f"寫入報告時發生錯誤：{e}")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🔍 現場線索輸入")
    uploaded_file = st.file_uploader("📸 拍照上傳現場照片 (支援 jpg, png)", type=["jpg", "jpeg", "png"])
    audio_file = st.audio_input("🎙️ 語音口述現場狀況：")
    user_text = st.text_area("⌨️ 現場狀況描述 (打字區)：", height=100)

system_instruction = """
你是一位嚴謹的電梯維修專家。請根據下方提供的「參考手冊與過往經驗紀錄」，檢視現場工程師提供的文字、語音或照片：
1. 找出並核對資料中對應的錯誤代碼、組件名稱或故障說明。
2. 條列出符合官方規範的排查步驟與現場維修安全守則。
3. ⚠️ 重要：如果參考資料中包含「過往維修報告」，請明確指出過往是否有類似案例、當時是如何解決的。
- 如果參考資料中完全沒有提及此狀況，請誠實告知，切勿自行瞎掰。
"""

with col2:
    st.markdown("### 📋 AI 專家檢修報告")
    if st.button("🚀 開始跨檔案對齊與診斷分析", use_container_width=True):
        if not (uploaded_file or audio_file or user_text.strip()):
            st.warning("⚠️ 請至少提供一項現場資訊！")
        elif db is None:
            st.error("知識庫未成功建立，請檢查左側狀態。")
        else:
            with st.spinner('正在跨手冊與維修紀錄調閱原文，比對分析中...'):
                try:
                    actual_problem = ""
                    if user_text.strip():
                        actual_problem += f"【工程師文字描述】：{user_text}\n"
                    
                    if audio_file:
                        st.toast("正在轉換現場語音...", icon="🎙️")
                        audio_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
                        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
                        audio_payload = {
                            "contents": [{
                                "parts": [
                                    {"inlineData": {"mimeType": "audio/wav", "data": encoded_audio}},
                                    {"text": "請精準辨識這段語音，直接輸出繁體中文逐字稿，不需任何額外說明。"}
                                ]
                            }]
                        }
                        audio_resp = requests.post(audio_url, headers={'Content-Type': 'application/json'}, json=audio_payload)
                        audio_resp.raise_for_status()
                        spoken_text = audio_resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        st.info(f"🗣️ 語音辨識結果：{spoken_text}")
                        actual_problem += f"【工程師語音描述】：{spoken_text}\n"

                    search_query = actual_problem if actual_problem.strip() else "請根據照片中的異常現象，提供可能的故障原因。"
                    relevant_docs = db.as_retriever(search_kwargs={"k": 4}).invoke(search_query)
                    
                    context_str = "\n\n--- 參考手冊與過往經驗紀錄 ---\n"
                    for i, doc in enumerate(relevant_docs):
                        context_str += f"[參考文本段落 {i+1} (來源: {os.path.basename(doc.metadata.get('source', '未知來源'))})]:\n{doc.page_content}\n"
                    
                    gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
                    final_parts = [
                        {"text": system_instruction},
                        {"text": actual_problem},
                        {"text": context_str}
                    ]
                    
                    if uploaded_file:
                        encoded_img = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                        final_parts.append({"inlineData": {"mimeType": uploaded_file.type, "data": encoded_img}})
                        st.image(uploaded_file, caption="上傳的現場照片", use_container_width=True)
                        
                    gen_payload = {"contents": [{"parts": final_parts}]}
                    gen_resp = requests.post(gen_url, headers={'Content-Type': 'application/json'}, json=gen_payload)
                    gen_resp.raise_for_status()
                    
                    response_text = gen_resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    
                    st.success("分析完成！")
                    st.write(response_text)
                    
                    with st.expander("🔍 檢視本次 AI 參考的知識庫原文片段"):
                        st.write(context_str)
                        
                except Exception as e:
                    st.error(f"分析時發生錯誤：{e}")
