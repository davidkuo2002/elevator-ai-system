import streamlit as st
import os
import time
import random
from google import genai
from PIL import Image

# 載入 PDF 處理與本機資料庫的最基礎工具
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ==========================================
# 🔐 安全設定區 (絕對不可寫死真實金鑰)
# ==========================================
# 系統會自動去 Streamlit 後台的 Secrets 尋找 GEMINI_API_KEY，保護您的密碼不外洩
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("⚠️ 系統偵測不到 API 金鑰！請確保已在 Streamlit 後台的 Secrets 中設定 `GEMINI_API_KEY`。")
    st.stop()

# 🕵️ 抓漏雷達：顯示金鑰前 4 碼與總長度，確認雲端到底讀到什麼（若正常運作後可自行刪除這兩行）
st.info(f"🕵️ 抓漏雷達：目前系統讀取到的金鑰開頭為 【 {API_KEY[:4]} 】")
st.info(f"📏 金鑰總長度為：{len(API_KEY)} 個字元")

# 自動鎖定 app.py 當前所在的資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定義存放官方手冊、歷史報告與資料庫的路徑
MANUALS_DIR = os.path.join(BASE_DIR, "manuals")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")  
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# 自動防呆：確保必要的資料夾都存在
os.makedirs(MANUALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# ==========================================

# 🚀 初始化 Google GenAI 客戶端 (特殊修正：加入強制標頭，破解新版 AQ 金鑰的 401 誤判 Bug)
client = genai.Client(
    api_key=API_KEY,
    http_options={'headers': {'x-goog-api-key': API_KEY}}
)
os.environ["GEMINI_API_KEY"] = API_KEY

# ------------------------------------------
# 特徵轉換包裝類別 (具備工業級指數退避重試與抗壓機制)
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
                    st.sidebar.warning(
                        f"⚠️ 觸發流量上限 (429)。系統將於 {delay:.1f} 秒後進行第 {attempt + 1} 次安全重試..."
                    )
                    time.sleep(delay)
                else:
                    raise e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 20  
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = self._embed_with_retry(
                model="gemini-embedding-001", 
                contents=batch_texts
            )
            all_embeddings.extend([embedding.values for embedding in response.embeddings])
            time.sleep(0.5)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self._embed_with_retry(
            model="gemini-embedding-001",
            contents=[text]
        )
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
with st.sidebar.spinner('正在檢查並喚醒雙軌知識庫中...'):
    db, error_message = setup_rag_database_multi_file()

if error_message:
    st.sidebar.error(error_message)
else:
    st.sidebar.success("官方手冊 + 歷史經驗知識庫已就緒！")

st.title("🛠️ 電梯維修 AI 智聯專家系統 (手冊與實戰經驗雙軌版)")
st.write("本系統已整合「官方技術手冊」與「前線維修報告」。請提供現場線索，AI 將自動對齊規範與過往實戰經驗進行診斷。")

# ==========================================
# 📥 知識庫經驗補給站
# ==========================================
with st.expander("📥 知識庫經驗補給站 (上傳過往維修報告、擴充 AI 大腦)", expanded=False):
    st.info("💡 **提示：** 當前線遇到特殊故障並排除後，請將結案報告(PDF)上傳至此。日後若有類似災情，AI 會自動調閱此紀錄給予建議。")
    
    st.markdown("""
    <div style="background-color:#f0f4f8; padding:15px; border-radius:8px; border-left:5px solid #2b5c8f; margin-bottom:15px;">
        <h5 style="margin-top:0; color:#2b5c8f; font-weight:bold;">📋 歷史報告命名標準規範</h5>
        <p style="margin-bottom:5px; font-size:14px;">為確保系統正確排序與檢索，請務必將 PDF 檔名修改為以下格式再上傳：</p>
        <code style="font-size:15px; font-weight:bold; color:#d9534f;">[日期]_[服務編號]_[客戶名稱]_[故障情形].pdf</code>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_report = st.file_uploader("請選擇要匯入的維修紀錄、結案報告 (限 PDF)", type=["pdf"], key="report_uploader")
    
    if uploaded_report is not None:
        filename_without_ext, ext = os.path.splitext(uploaded_report.name)
        parts = filename_without_ext.split('_')
        is_format_valid = len(parts) == 4 and all(p.strip() for p in parts)
        
        if not is_format_valid:
            st.error(f"❌ <b>檔名格式錯誤！</b>請重新命名為 `[日期]_[服務編號]_[客戶名稱]_[故障情形].pdf` 後再重新上傳。", icon="🚫")
        else:
            st.success("✅ <b>檔名格式檢查通過！</b>", icon="🔥")
            save_path = os.path.join(REPORTS_DIR, uploaded_report.name)
            
            if os.path.exists(save_path):
                st.warning(f"ℹ️ 檔案 `{uploaded_report.name}` 先前已匯入過，無需重複上傳。")
            else:
                if st.button("確認將此報告寫入本機知識庫", use_container_width=True):
                    with st.spinner("正在解析報告並進行特徵向量增量寫入..."):
                        try:
                            with open(save_path, "wb") as f:
                                f.write(uploaded_report.getbuffer())
                            
                            loader = PyPDFLoader(save_path)
                            new_docs = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                            new_splits = text_splitter.split_documents(new_docs)
                            
                            if db is not None:
                                db.add_documents(new_splits)
                                st.success(f"🎉 成功！經驗已動態融入大腦。")
                                time.sleep(1.5)
                                st.rerun() 
                            else:
                                st.error("知識庫本體尚未建立，無法寫入.。。")
                        except Exception as e:
                            st.error(f"寫入報告時發生錯誤：{e}")

st.markdown("---")

# ==========================================
# 現場檢修診斷區
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🔍 現場線索輸入")
    uploaded_file = st.file_uploader("📸 拍照上傳現場照片 (支援 jpg, png)", type=["jpg", "jpeg", "png"])
    audio_file = st.audio_input("🎙️ 語音口述現場狀況：")
    user_text = st.text_area(
        "⌨️ 現場狀況描述 (打字區)：", 
        placeholder="例如：主機發出尖銳異音，面板顯示 E-02，車廂停在五樓...", 
        height=100
    )

system_instruction = """
你是一位嚴謹的電梯維修專家。請根據下方提供的「參考手冊與過往經驗紀錄」，檢視現場工程師提供的文字、語音或照片：
1. 找出並核對資料中對應的錯誤代碼、組件名稱或故障說明。
2. 條列出符合官方規範的排查步驟與現場維修安全守則。
3. ⚠️ 重要：如果參考資料中包含「過往維修報告（或歷史紀錄）」，請明確指出過往是否有類似案例、當時是如何解決的，並給予工程師實戰檢修建議。
- 如果參考資料中完全沒有提及此狀況，請誠實告知「手冊與過往紀錄中未記載此項資訊」，切勿自行瞎掰。
"""

with col2:
    st.markdown("### 📋 AI 專家檢修報告")
    if st.button("🚀 開始跨檔案對齊與診斷分析", use_container_width=True):
        has_image = uploaded_file is not None
        has_audio = audio_file is not None
        has_text = bool(user_text.strip())
        
        if not (has_image or has_audio or has_text):
            st.warning("⚠️ 請至少提供一項現場資訊！")
        elif db is None:
            st.error("知識庫未成功建立，請檢查左側狀態。")
        else:
            with st.spinner('正在跨手冊與維修紀錄調閱原文，比對分析中...'):
                try:
                    actual_problem = ""
                    if has_text:
                        actual_problem += f"【工程師文字描述】：{user_text}\n"
                    
                    if has_audio:
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
                    retriever = db.as_retriever(search_kwargs={"k": 4}) 
                    relevant_docs = retriever.invoke(search_query)
                    
                    context_str = "\n\n--- 參考手冊與過往經驗紀錄 ---\n"
                    for i, doc in enumerate(relevant_docs):
                        context_str += f"[參考文本段落 {i+1} (來源: {os.path.basename(doc.metadata.get('source', '未知來源'))})]:\n{doc.page_content}\n"
                    context_str += "----------------------\n"
                    
                    full_content = [system_instruction]
                    if actual_problem.strip():
                        full_content.append(actual_problem)
                    full_content.append(context_str)
                    
                    if has_image:
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
