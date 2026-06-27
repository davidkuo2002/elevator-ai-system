import streamlit as st
import os
from google import genai
from PIL import Image

# 載入 PDF 處理與本機資料庫的最基礎工具
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ==========================================
# 填寫設定區
# ==========================================
# 請在此處貼上您以 AQ. 開頭的真實 API Key（務必保留雙引號）
API_KEY = "AQ.Ab8RN6LgosrQ8M67Qvez22elmXqAzjwxiXT0MxbWITVzgCJcpA"

# 自動鎖定 app.py 當前所在的資料夾絕對路徑（徹底解決找不到檔案的問題）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = os.path.join(BASE_DIR, "manual.pdf")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
# ==========================================

# 初始化 Google GenAI 客戶端
client = genai.Client(api_key=API_KEY)
os.environ["GEMINI_API_KEY"] = API_KEY

# ------------------------------------------
# 建立一個安全的特徵轉換包裝類別，繞過 LangChain 版本衝突
# ------------------------------------------
class GenAIEmbeddingsWrapper:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = client.models.embed_content(model="text-embedding-004", contents=texts)
        return [embedding.values for embedding in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = client.models.embed_content(model="text-embedding-004", contents=text)
        return response.embeddings[0].values

@st.cache_resource
def setup_rag_database():
    if not os.path.exists(PDF_FILE):
        return None, f"找不到手冊檔案！請確認手冊檔名為 manual.pdf，且已放在資料夾中。\n目前偵測路徑為：{PDF_FILE}"
    
    try:
        # 1. 讀取 PDF
        loader = PyPDFLoader(PDF_FILE)
        documents = loader.load()
        
        # 2. 切割文字（每塊 600 字，重疊 50 字）
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        # 3. 寫入本機 Chroma 資料庫
        embeddings = GenAIEmbeddingsWrapper()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
        return vectorstore, None
    except Exception as e:
        return None, f"後台建立知識庫時發生錯誤：{e}"

# 執行側邊欄後台初始化
st.sidebar.markdown("### 📚 知識庫後台狀態")
with st.sidebar.spinner('正在翻閱並研讀電梯維修手冊，請稍候...'):
    db, error_message = setup_rag_database()

if error_message:
    st.sidebar.error(error_message)
else:
    st.sidebar.success("官方維修手冊研讀完畢！AI 已就緒。")

# ------------------------------------------
# 前端網頁介面
# ------------------------------------------
st.set_page_config(page_title="電梯維修 AI 助手 (專業手冊版)", page_icon="🛠️")
st.title("🛠️ 電梯維修 AI 助手 (專業手冊版)")
st.write("上傳電梯異常現場照片，AI 將自動調閱您提供的官方手冊進行精準對齊診斷。")

uploaded_file = st.file_uploader("請上傳現場照片 (支援 jpg, png)", type=["jpg", "jpeg", "png"])

default_prompt = """
你是一位嚴謹的電梯維修專家。請根據下方提供的「參考手冊內容」，檢視這張現場照片：
1. 找出並核對手冊中對應的錯誤代碼或故障說明。
2. 條列出符合官方規範的排查步驟與現場維修安全守則。
如果參考手冊內容中完全沒有提及此狀況，請誠實告知「手冊中未記載此項資訊」，切勿自行瞎掰。
"""
user_prompt = st.text_area("補充說明或工程師提問：", value=default_prompt, height=150)

if st.button("🚀 開始手冊比對與分析"):
    if uploaded_file is not None and db is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="上傳的現場照片", use_container_width=True)
        
        with st.spinner('正在調閱手冊原文並進行對比分析中...'):
            try:
                # 1. 檢索：拿工程師的提問去資料庫撈最相關的 3 段手冊內容
                retriever = db.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_prompt)
                
                # 2. 組合上下文
                context_str = "\n\n--- 參考手冊內容 ---\n"
                for i, doc in enumerate(relevant_docs):
                    context_str += f"[手冊參考段落 {i+1}]:\n{doc.page_content}\n"
                context_str += "----------------------\n"
                
                # 3. 送出給最新 Gemini 模型
                full_content = [user_prompt, context_str, img]
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=full_content
                )
                
                st.success("分析完成！")
                st.markdown("### 📋 依據官方手冊之診斷報告")
                st.write(response.text)
                
                # 供技師查核比對的摺疊區塊
                with st.expander("🔍 檢視本次 AI 參考的手冊原文片段"):
                    st.write(context_str)
                    
            except Exception as e:
                st.error(f"分析時連線發生錯誤：{e}")
    elif uploaded_file is None:
        st.warning("⚠️ 請先上傳一張現場照片！")
    elif db is None:
        st.error("知識庫未成功建立，請檢查左側後台錯誤訊息。")
