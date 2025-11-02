"""
HDDHT Messenger Bot - Tích hợp đầy đủ
Tự động nhận tin nhắn -> Xử lý với RAG -> Trả lời
"""

import requests
import time
from datetime import datetime, timedelta
import sys
import os
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

# Config
PAGE_ACCESS_TOKEN = "EAANZB8eGZACqoBPyScnYP7qbn3e9jurB38QMV7infbF3lniZCMfIwpIRq03kZBIuT0ws75lGO4RwPsUxfEcg5EkDZB3EgViD1ptB5VPs9vEjyWlzX9ue9HouwZCOAR0EWXGkML8ZByo4IeuGycglXfnPgZAubTNvZCIBkJiCbaYLqOWzpP7P0ByXy2j7KP4oRrkbByGs40QZDZD"
DOCUMENT_PATH = r"C:\Users\mduc1\OneDrive\Desktop\HDDT\Tài liệu không có tiêu đề.txt"
POLL_INTERVAL = 8  # Check every 8 seconds

# Biến global
MY_PAGE_ID = None
SEEN_MESSAGES = set()
USER_CACHE = {}
START_TIME = None


# Chatbot RAG fusion Finetune
class HDDHTAssistant:
    def __init__(self, document_path: str, persist_directory: str = "./chroma_db"):
        self.document_path = document_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
        print("🔧 Đang khởi tạo AI model...")
        self.llm = Ollama(
            model="deepseek-r1:8b",
            temperature=0.2,
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def load_and_split_document(self):
        print("📄 Đang đọc tài liệu...")
        loader = TextLoader(self.document_path, encoding='utf-8')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"✅ Đã chia tài liệu thành {len(splits)} đoạn văn")
        return splits
    
    def create_vectorstore(self):
        print("🔄 Đang tạo vector database...")
        splits = self.load_and_split_document()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Vector database đã được lưu")
        
    def load_vectorstore(self):
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"Không tìm thấy vector database")
        
        print("📂 Đang load vector database...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("✅ Đã load vector database thành công")
    
    def setup_qa_chain(self):
        if self.vectorstore is None:
            raise ValueError("Vector database chưa được khởi tạo")
        
        prompt_template = """Bạn là trợ lý tư vấn chuyên về hóa đơn điện tử (HĐĐT) tại Việt Nam. 
Nhiệm vụ của bạn là trả lời câu hỏi DỰA TRÊN CHÍNH XÁC thông tin trong tài liệu được cung cấp.

NGUYÊN TẮC:
1. CHỈ trả lời dựa trên thông tin trong Context bên dưới
2. Nếu không tìm thấy thông tin, trả lời: "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu."
3. Trả lời bằng tiếng Việt, súc tích, thân thiện
4. Trích dẫn số liệu cụ thể nếu có (chi phí, thời gian, mẫu biểu)

Context từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ Chatbot đã sẵn sàng!")
    
    def ask(self, question: str):
        """Hỏi chatbot và trả về câu trả lời"""
        if self.qa_chain is None:
            raise ValueError("QA chain chưa được setup")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return result['result'].strip()
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý câu hỏi: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."


# Get info and message 
def get_my_page_id():
    """Lấy Page ID"""
    global MY_PAGE_ID
    
    url = "https://graph.facebook.com/v21.0/me"
    params = {"access_token": PAGE_ACCESS_TOKEN}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            MY_PAGE_ID = response.json().get('id')
            print(f"✅ Page ID: {MY_PAGE_ID}")
            return MY_PAGE_ID
        elif response.status_code == 401:
            print("❌ Lỗi xác thực: Token không hợp lệ hoặc đã hết hạn!")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi kết nối: {e}")
        return None
    
    print("❌ Không thể lấy Page ID!")
    return None


def get_user_info(user_id):
    """Lấy thông tin người dùng"""
    if user_id in USER_CACHE:
        return USER_CACHE[user_id]
    
    url = f"https://graph.facebook.com/v21.0/{user_id}"
    params = {
        "access_token": PAGE_ACCESS_TOKEN,
        "fields": "name,first_name"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            name = data.get('name', 'Unknown User')
            USER_CACHE[user_id] = name
            return name
    except requests.exceptions.RequestException:
        pass
    
    return "Unknown User"


def send_message(recipient_id, message_text):
    """Gửi tin nhắn đến user"""
    url = "https://graph.facebook.com/v21.0/me/messages"
    params = {"access_token": PAGE_ACCESS_TOKEN}
    
    data = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    
    try:
        response = requests.post(url, params=params, json=data, timeout=10)
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Lỗi gửi tin nhắn: {response.json()}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi kết nối khi gửi tin nhắn: {e}")
        return False


def check_new_messages():
    """Kiểm tra tin nhắn mới"""
    url = "https://graph.facebook.com/v21.0/me/conversations"
    params = {
        "access_token": PAGE_ACCESS_TOKEN,
        "fields": "unread_count,messages.limit(20){message,from{id,name},created_time,id,sticker,to}",
        "limit": 100
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 401:
            print("\n⚠️  TOKEN HẾT HẠN!")
            sys.exit(1)
        
        if response.status_code != 200:
            return []
        
        conversations = response.json().get('data', [])
        new_messages = []
        
        for convo in conversations:
            unread_count = convo.get('unread_count', 0)
            if unread_count == 0:
                continue
            
            messages = convo.get('messages', {}).get('data', [])
            unread_msgs = messages[:unread_count]
            
            for msg in unread_msgs:
                msg_id = msg.get('id')
                
                if msg_id in SEEN_MESSAGES:
                    continue
                
                created_time = msg.get('created_time')
                if created_time:
                    msg_time = datetime.fromisoformat(created_time.replace('Z', '+00:00')).replace(tzinfo=None)
                    if msg_time < START_TIME:
                        SEEN_MESSAGES.add(msg_id)
                        continue
                
                from_info = msg.get('from', {})
                from_id = from_info.get('id')
                
                if from_id == MY_PAGE_ID:
                    continue
                
                user_name = from_info.get('name', None)
                if not user_name:
                    user_name = get_user_info(from_id)
                else:
                    USER_CACHE[from_id] = user_name
                
                SEEN_MESSAGES.add(msg_id)
                
                if msg.get('message') or not msg.get('sticker'):
                    new_messages.append({
                        'user_id': from_id,
                        'user_name': user_name,
                        'message': msg.get('message', '(Tin nhắn không có nội dung)'),
                        'created_time': created_time,
                        'msg_id': msg_id
                    })
        
        new_messages.sort(key=lambda x: x['created_time'])
        return new_messages
    
    except Exception as e:
        print(f"\n⚠️  Lỗi: {e}")
        return []


def format_time(iso_time):
    """Format thời gian"""
    try:
        dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        dt_vn = dt + timedelta(hours=7)
        return dt_vn.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return iso_time


# Main Loop
def main():
    global START_TIME
    
    print("\n" + "╔" + "=" * 88 + "╗")
    print("║" + " " * 15 + "🤖 HDDHT MESSENGER BOT - AUTO REPLY SYSTEM" + " " * 30 + "║")
    print("╚" + "=" * 88 + "╝")
    
    # Chatbot initialization
    print("\n📦 BƯỚC 1: Khởi tạo AI Chatbot...")
    assistant = HDDHTAssistant(DOCUMENT_PATH)
    
    if os.path.exists("./chroma_db"):
        print("📂 Phát hiện vector database có sẵn")
        assistant.load_vectorstore()
    else:
        assistant.create_vectorstore()
    
    assistant.setup_qa_chain()
    
    # Khởi tạo Messenger
    print("\n📱 BƯỚC 2: Kết nối Facebook Messenger...")
    if not get_my_page_id():
        sys.exit(1)
    
    START_TIME = datetime.utcnow().replace(tzinfo=None)
    start_time_vn = START_TIME + timedelta(hours=7)
    
    print(f"⏱️  Chu kỳ kiểm tra: {POLL_INTERVAL} giây")
    print(f"🕐 Bắt đầu từ: {start_time_vn.strftime('%d/%m/%Y %H:%M:%S')} (Giờ VN)")
    print("🟢 Bot đang hoạt động... (nhấn Ctrl+C để dừng)\n")
    
    message_count = 0
    
    try:
        while True:
            new_msgs = check_new_messages()
            
            if new_msgs:
                for msg in new_msgs:
                    message_count += 1
                    
                    # Hiển thị tin nhắn nhận được
                    print(f"\n{'═' * 90}")
                    print(f"📩 TIN NHẮN #{message_count}")
                    print(f"{'═' * 90}")
                    print(f"👤 Người gửi: {msg['user_name']}")
                    print(f"💬 Câu hỏi:  {msg['message']}")
                    print(f"🕐 Thời gian: {format_time(msg['created_time'])}")
                    
                    # Xử lý với AI
                    print(f"🤔 Đang suy nghĩ...")
                    answer = assistant.ask(msg['message'])
                    
                    print(f"💡 Câu trả lời: {answer[:100]}...")
                    
                    # Gửi phản hồi
                    print(f"📤 Đang gửi phản hồi...")
                    if send_message(msg['user_id'], answer):
                        print(f"✅ Đã gửi thành công!")
                    else:
                        print(f"❌ Gửi thất bại!")
                    
                    print(f"{'═' * 90}")
            
            time.sleep(POLL_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n" + "╔" + "=" * 88 + "╗")
        print("║" + " " * 35 + "⛔ DỪNG BOT" + " " * 42 + "║")
        print(f"║" + " " * 20 + f"Tổng tin nhắn đã xử lý: {message_count}" + " " * 34 + "║")
        print("╚" + "=" * 88 + "╝\n")
        sys.exit(0)


if __name__ == "__main__":
    main()