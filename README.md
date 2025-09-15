# DocTalk AI ✨

DocTalk AI is an AI-powered application that enables interactive conversations with your documents.
Simply upload your files (PDF, DOCX, PPTX, XLSX, TXT, or MD), and ask questions in natural language.
The system retrieves relevant content using FAISS vector search and provides accurate, context-based answers through Google Gemini AI.
Optional voice output lets you listen to responses with natural-sounding speech using gTTS.

✨ Features
📂 Multi-format Document Support – PDF, Word, PowerPoint, Excel, Text, Markdown
🔎 Document Search & Retrieval – Uses FAISS + Google AI Embeddings
💬 Conversational AI – Powered by Gemini (Generative AI)
🎙️ Voice Output – Converts answers into speech with gTTS
🗂️ Sidebar File Management – Upload multiple files at once
📎 In-Chat File Upload – Attach files directly while chatting
💾 Download Conversation – Export your entire chat history as text
🎨 Clean Gemini-like UI – Light & dark theme adaptive styling

🛠️ Tech Stack
Frontend: Streamlit
LLM: Google Gemini (gemini-1.5-flash-latest)
Embeddings: GoogleGenerativeAIEmbeddings (LangChain)
Vector Store: FAISS
Document Parsing: PyPDF2, python-docx, python-pptx, openpyxl
Text-to-Speech: gTTS
Utilities: asyncio, base64

📂 Supported File Types
.pdf (PDF documents)
.docx (Word documents)
.pptx (PowerPoint presentations)
.xlsx (Excel spreadsheets)
.txt (Plain text)
.md (Markdown files)
