🤖 Hybrid Rule-Based + Wikipedia Chatbot

A simple yet powerful hybrid chatbot built with Python that combines a local rule-based knowledge base with Wikipedia search to answer user questions intelligently.
If a question is not found locally, the bot automatically fetches a short summary from Wikipedia.

📌 Features

✅ Local Knowledge Base (Offline Q&A)

🔍 Fuzzy Matching for similar questions using difflib

🌐 Wikipedia Integration for dynamic answers

🔡 Case-insensitive input handling

🧠 Hybrid decision pipeline (Local → Wikipedia → Fallback)

📝 Optional Chat Logging with timestamps

⚠️ Graceful handling of Wikipedia errors

🐍 Compatible with Python 3.8+

🛠️ Technologies Used

Python 3.8+

wikipedia library

difflib

Regular Expressions (re)

File handling & datetime utilities

📂 Project Structure
hybrid-chatbot/
│
├── chatbot.py          # Main chatbot program
├── chat_history.txt    # Auto-generated chat log (optional)
└── README.md           # Project documentation

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Abdulwasaytahir/hybrid-chatbot.git
cd hybrid-chatbot

2️⃣ Install Required Library
pip install wikipedia

3️⃣ Run the Chatbot
python chatbot.py

💬 How It Works

User Input

Quick Rule-Based Replies (greetings, help, exit, etc.)

Fuzzy Match Search in local knowledge base

Wikipedia Lookup if local data fails

Fallback Response if nothing is found

📘 Example Interaction
You: What is Python?
ChatBot: Python is a high-level, interpreted programming language known for its readability and large ecosystem.

You: Who is Albert Einstein?
ChatBot: According to Wikipedia (Albert Einstein):
Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.

🧩 Local Knowledge Base

You can easily expand the offline knowledge by editing the dictionary:

QA_KB = {
    "what is python": "Python is a high-level, interpreted programming language...",
    "what is ai": "AI stands for Artificial Intelligence...",
}

⚙️ Configuration Options
FUZZY_CUTOFF = 0.6        # Matching accuracy
WIKI_SENTENCES = 2       # Wikipedia summary length
ENABLE_CHAT_LOG = True   # Enable/disable chat logging

📄 Chat Logging

When enabled, all conversations are saved to:

chat_history.txt


Each entry includes a timestamp for both user and bot messages.

🧪 Error Handling

The chatbot safely handles:

Wikipedia page not found

Disambiguation errors

Network-related issues

Empty or invalid input

🎯 Future Improvements

NLP using spaCy or NLTK

GUI (Tkinter / Web Interface)

Voice-based input/output

Database-backed knowledge base

Multilingual support

👤 Author

Abdul Wasay Tahir
💡 Computer Science & Robotics Enthusiast
🚀 Aspiring Software Engineer

📜 License

This project is open-source and free to use for learning, modification, and personal projects.
