import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid
from ollama import Client

# 页面配置
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式 - 模仿DeepSeek风格
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2E86AB;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .message-header {
        font-weight: 600;
        margin-bottom: 5px;
        color: #555;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #888;
        text-align: right;
    }
    .stButton button {
        background-color: #2E86AB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1a5a7a;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .conversation-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .conversation-item:hover {
        background-color: #e3f2fd;
    }
    .conversation-item.active {
        background-color: #2E86AB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class ChatManager:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

        # OLLAMA 代理
        self.OLLAMA_MODEL = 'qwen3:0.6b'

        self.client = Client(
            host='http://localhost:11434'
        )

        if 'conversations' not in st.session_state:
            st.session_state.conversations = {}
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = None
        if 'message_id' not in st.session_state:
            st.session_state.message_id = 0

    def create_new_conversation(self):
        """创建新的对话"""
        conv_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[conv_id] = {
            'id': conv_id,
            'title': f"对话 {len(st.session_state.conversations) + 1}",
            'messages': [],
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.current_conversation = conv_id
        return conv_id

    def add_message(self, role, content):
        """添加消息到当前对话"""
        if st.session_state.current_conversation:
            message = {
                'id': st.session_state.message_id,
                'role': role,
                'content': content,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.conversations[st.session_state.current_conversation]['messages'].append(message)
            st.session_state.message_id += 1

    def call_ollama(self, prompt):
        """调用本地Ollama模型"""
        try:
            # payload = {
            #     "model": "qwen3:0.6b",  # 你可以更改为你安装的模型
            #     "prompt": prompt,
            #     "stream": False
            # }
            # response = requests.post(self.ollama_url, json=payload, timeout=30)

            response = self.client.generate(model=self.OLLAMA_MODEL, 
                                    prompt=prompt,
                                    options={'temperature': 1.0, 'top_p': 1.0})
            return response.response
        except requests.exceptions.RequestException as e:
            return f"连接错误: {str(e)}"

    def get_conversation_history(self):
        """获取当前对话的历史"""
        if st.session_state.current_conversation:
            return st.session_state.conversations[st.session_state.current_conversation]['messages']
        return []

def main():
    chat_manager = ChatManager()
    
    # 标题
    st.markdown('<div class="main-header">🤖 创典智能 AI 助手</div>', unsafe_allow_html=True)
    
    # 侧边栏 - 对话管理
    with st.sidebar:
        st.header("💬 对话管理")
        
        # 新建对话按钮
        if st.button("➕ 新建对话", use_container_width=True):
            chat_manager.create_new_conversation()
            st.rerun()
        
        st.markdown("---")
        st.subheader("历史对话")
        
        # 显示对话列表
        for conv_id, conv in st.session_state.conversations.items():
            is_active = conv_id == st.session_state.current_conversation
            item_class = "conversation-item active" if is_active else "conversation-item"
            
            if st.button(f"💬 {conv['title']}", key=conv_id, use_container_width=True):
                st.session_state.current_conversation = conv_id
                st.rerun()
            
            # 显示对话统计信息
            if is_active:
                st.caption(f"消息数: {len(conv['messages'])}")
                st.caption(f"创建于: {conv['created_at']}")
        
        st.markdown("---")
        st.subheader("模型设置")
        
        # 模型选择（这里简化处理，实际使用时可以动态获取可用的模型）
        model_options = ["qwen3:0.6b", "gemma3:1b", "qwen2.5-coder:1.5b", "qwen2.5-coder:0.5b"]
        chat_manager.OLLAMA_MODEL = st.selectbox("选择模型", model_options, index=0)
        
        st.info("确保Ollama服务在localhost:11434运行")

    # 主聊天区域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💭 AI 对话")
        
        # 显示当前对话的消息
        if st.session_state.current_conversation:
            messages = chat_manager.get_conversation_history()
            
            for msg in messages:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="message-header">👤 你</div>
                        <div>{msg['content']}</div>
                        <div class="timestamp">{msg['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="message-header">🤖 AI助手</div>
                        <div>{msg['content']}</div>
                        <div class="timestamp">{msg['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 点击侧边栏的'新建对话'开始聊天！")

    with col2:
        st.subheader("⚡ 快速操作")
        
        if st.button("🔄 刷新对话"):
            st.rerun()
            
        if st.button("🗑️ 清空当前对话") and st.session_state.current_conversation:
            st.session_state.conversations[st.session_state.current_conversation]['messages'] = []
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 统计信息")
        if st.session_state.current_conversation:
            conv = st.session_state.conversations[st.session_state.current_conversation]
            st.metric("消息总数", len(conv['messages']))
            st.metric("当前对话", conv['title'])
        else:
            st.metric("消息总数", 0)
            st.metric("当前对话", "无")

    # 输入区域
    st.markdown("---")
    input_col1, input_col2 = st.columns([4, 1])
    
    with input_col1:
        user_input = st.text_area(
            "💬 输入你的消息:",
            placeholder="在这里输入你的问题...",
            height=100,
            key="user_input"
        )
    
    with input_col2:
        st.write("")  # 垂直间距
        st.write("")
        send_button = st.button("🚀 发送", use_container_width=True)


    # 添加简化的JavaScript监听
    st.components.v1.html("""
    <script>
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            // 找到发送按钮并点击
            const buttons = window.parent.document.querySelectorAll('button');
            for (let btn of buttons) {
                if (btn.innerText === '🚀 发送') {
                    btn.click();
                    break;
                }
            }
        }
    });
    </script>
    """, height=0)
    
    # 处理用户输入
    if send_button and user_input.strip():
        if not st.session_state.current_conversation:
            chat_manager.create_new_conversation()
        
        # 添加用户消息
        chat_manager.add_message('user', user_input.strip())
        
        # 显示思考中的消息
        with st.spinner("🤔 AI正在思考中..."):
            # 调用Ollama模型
            response = chat_manager.call_ollama(user_input.strip())
            
            # 添加AI回复
            # chat_manager.add_message('assistant', response)
            chat_manager.add_message('assistant', response)
        
        st.rerun()

if __name__ == "__main__":
    main()