# HDDHT Messenger Bot - E-Invoice Consulting Chatbot

> **Automatically receive Facebook messages → Process with AI RAG → Answer accurately based on documents**

---

## Key Features

- **24/7 automatic replies** on **Facebook Messenger**
- **AI RAG (Retrieval-Augmented Generation)** – answers **based on documents only**
- Uses **Ollama + deepseek-r1:8b** + **Vietnamese SBERT**
- **No webhook required** – works with **polling (every 8 seconds)**
- Beautiful console interface, easy to monitor
- Stores vector DB (`Chroma`) for reuse

---

## System Requirements

| Requirement | Version |
|------------|---------|
| Python | `3.9+` |
| Ollama | Installed + model `deepseek-r1:8b` |
| RAM | Minimum 8GB (for 8B model) |
| OS | Windows / macOS / Linux |

---

## Quick Setup (5 minutes)

### Step 1: Install Ollama & Model

```bash
# Install Ollama: https://ollama.com/download
ollama pull deepseek-r1:8b
```

### Step 2: Install Python Dependencies

```bash
pip install requests langchain langchain-community chromadb sentence-transformers ollama
```

### Step 3: Install and Run ngrok (Optional - for webhook mode)

```bash
# Download ngrok: https://ngrok.com/download
# Run ngrok to expose your local server
ngrok http 5000
```

**Note**: This bot uses **polling mode** by default, so ngrok is **not required**. However, if you want to use webhook mode for faster response times, you'll need ngrok to expose your local server to the internet.

### Step 4: Prepare Documents

- Place your document `.txt` file (UTF-8) on your machine
- Example: `E-Invoice Documentation.txt`

---

## Configuration & Running

### 1. Edit `Chatbot_HDDT.py`

```python
PAGE_ACCESS_TOKEN = "PASTE_YOUR_TOKEN_HERE"
DOCUMENT_PATH = r"C:\Path\To\Your\Document.txt"
```

**Where to get the token?**
→ Meta for Developers → App → Messenger → Access Tokens → Generate Token (Page)

### 2. Run the Bot

```bash
python Chatbot_HDDT.py
```

- **First time**: Creates vector DB from documents (~1-3 minutes)
- **Next times**: Loads quickly (<10s)

---

## Interface When Running

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                 HDDHT MESSENGER BOT - AUTO REPLY SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Initializing AI model...
Document split into 156 chunks
Vector database saved
Chatbot is ready!

Connecting to Facebook Messenger...
Page ID: 123456789
Check interval: 8 seconds
Started at: 11/03/2025 14:30:25 (Vietnam Time)
Bot is running... (press Ctrl+C to stop)

══════════════════════════════════════════════════════════════════════════════════════════
MESSAGE #1
══════════════════════════════════════════════════════════════════════════════════════════
Sender: John Doe
Question: How to cancel an e-invoice?
Time: 11/03/2025 14:31:10
Processing...
Answer: According to regulations, to cancel an e-invoice you need to create a cancellation report...
Sending response...
Successfully sent!
══════════════════════════════════════════════════════════════════════════════════════════
```

---

## Update Documents?

1. Delete the `chroma_db` folder
2. Run the bot again → automatically recreates vector DB

---

## Stop the Bot

Press `Ctrl + C`

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                    STOPPING BOT                                      ║
║                        Total messages processed: 12                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Usage Tips

| Purpose | Solution |
|---------|----------|
| Change AI model | Edit `model="deepseek-r1:8b"` → `llama3.2`, `gemma2`, etc. |
| Increase speed | Use `search_kwargs={"k": 2}` (fewer retrieved chunks) |
| Increase accuracy | Increase `chunk_size=1000`, `k=5` |
| Run in background | Use `screen` (Linux) or Task Scheduler (Windows) |

---

## Important Notes

- **Token expired?** → Generate new token → replace in code
- **Not receiving messages?** → Check if Page has messaging permissions
- **Bot slow?** → Use smaller model (`phi3`, `gemma2:2b`) or upgrade hardware

---

## Author

**HDDHT** –
Contact: mduc11011@gmail.com
