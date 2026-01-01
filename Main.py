"""
Hybrid Rule-Based + Wikipedia Chatbot
Author: Abdul Wasay Tahir

Features:
- Local knowledge base (dictionary of Q&A)
- Fuzzy matching for similar questions
- If the question is not found locally, tries to fetch a short summary from Wikipedia
- Case-insensitive, simple conversation loop
- Optional chat logging to a file (chat_history.txt)
- Handles common wikipedia errors gracefully
- Works with Python 3.8+
"""

import difflib
import wikipedia       # pip install wikipedia
import re
import datetime

# -----------------------------
# Local knowledge base (editable)
# -----------------------------
QA_KB = {
    "what is python": "Python is a high-level, interpreted programming language known for its readability and large ecosystem.",
    "who is elon musk": "Elon Musk is an entrepreneur and CEO of Tesla and SpaceX, among other ventures.",
    "what is ai": "AI stands for Artificial Intelligence — the simulation of human intelligence in machines.",
    "what is github": "GitHub is a platform for hosting and collaborating on software projects using Git.",
    "how to learn programming": "Start with fundamentals (variables, loops, functions), build small projects, and practice consistently.",
    "what is machine learning": "Machine Learning is a subset of AI that allows systems to learn patterns from data and make predictions.",
    # add more Q/A pairs here to enlarge the offline knowledge base
}

# -----------------------------
# Settings
# -----------------------------
FUZZY_CUTOFF = 0.6        # similarity threshold (0.0 - 1.0)
WIKI_SENTENCES = 2       # how many sentences to return from Wikipedia
ENABLE_CHAT_LOG = True   # set False to disable writing chat_history.txt

# -----------------------------
# Helpers
# -----------------------------
def normalize_text(text: str) -> str:
    """Lowercase + remove punctuation for better matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # collapse whitespace
    return text

def find_best_local_answer(user_q: str, kb: dict, cutoff: float = FUZZY_CUTOFF):
    """
    Try to find the best match in local knowledge base using difflib.
    Returns (match_key, answer, score) or (None, None, 0).
    """
    user_q_norm = normalize_text(user_q)
    keys = list(kb.keys())
    # compute close matches (difflib uses a quick heuristic)
    matches = difflib.get_close_matches(user_q_norm, keys, n=3, cutoff=cutoff)
    if matches:
        best = matches[0]
        # approximate score via SequenceMatcher
        score = difflib.SequenceMatcher(None, user_q_norm, best).ratio()
        return best, kb[best], score
    # fallback: also try substring checks
    for k in keys:
        if k in user_q_norm or user_q_norm in k:
            score = difflib.SequenceMatcher(None, user_q_norm, k).ratio()
            if score >= cutoff:
                return k, kb[k], score
    return None, None, 0.0

def fetch_wikipedia_summary(query: str, sentences: int = WIKI_SENTENCES):
    """
    Fetch a short summary from Wikipedia for the query.
    Returns a string or None if not available.
    """
    try:
        # try search first to get a good title
        search_results = wikipedia.search(query, results=5)
        if not search_results:
            return None
        # take the top result (most relevant)
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False, redirect=True)
        return f"According to Wikipedia ({page_title}):\n{summary}"
    except wikipedia.DisambiguationError as e:
        # choose first option from disambiguation if possible
        try:
            option = e.options[0]
            summary = wikipedia.summary(option, sentences=sentences, auto_suggest=False, redirect=True)
            return f"According to Wikipedia ({option}):\n{summary}"
        except Exception:
            return None
    except wikipedia.PageError:
        return None
    except Exception:
        return None

def log_chat(user_text: str, bot_text: str, logfile: str = "chat_history.txt"):
    """Append chat exchange to a logfile with timestamp."""
    if not ENABLE_CHAT_LOG:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] You: {user_text}\n")
        f.write(f"[{timestamp}] Bot: {bot_text}\n\n")

# -----------------------------
# Chatbot reply pipeline
# -----------------------------
def chatbot_reply(user_input: str) -> str:
    user_input = user_input.strip()
    if user_input == "":
        return "Please type something — I didn't get your message."

    # simple keyword-based quick replies
    low = user_input.lower()
    if any(g in low for g in ["hi", "hello", "hey"]):
        return "Hello! How can I help you today?"
    if "how are you" in low or "how r you" in low or "how are u" in low:
        return "I'm a program, doing fine! How about you?"
    if "your name" in low or "who are you" in low:
        return "I'm ChatBot — a hybrid assistant (local + Wikipedia). Type 'help' to see options."
    if low.strip() in ("thanks", "thank you", "thx"):
        return "You're welcome!"
    if low.strip() in ("bye", "goodbye", "exit", "quit"):
        return "Goodbye! Have a great day"

    if low.strip() == "help":
        return ("I can answer common questions from my local knowledge base. "
                "If I don't know, I'll try Wikipedia. Examples:\n"
                "- Who is Elon Musk?\n- What is Python?\nType 'exit' to leave.")

    # 1) Try local knowledge base (fuzzy)
    match_key, answer, score = find_best_local_answer(user_input, QA_KB)
    if answer:
        # Optionally show that it's from local KB when score is low-ish
        if score < 0.8:
            return f"(Local) {answer}"
        else:
            return f"{answer}"

    # 2) Not in KB -> try Wikipedia
    wiki_ans = fetch_wikipedia_summary(user_input)
    if wiki_ans:
        return wiki_ans

    # 3) Last fallback: generic reply
    return "Sorry, I couldn't find an answer. Try rephrasing or ask something else."

# -----------------------------
# Main conversational loop
# -----------------------------
def main():
    print("Hybrid ChatBot: Hello! (type 'help' for tips, 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChatBot: Bye!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("ChatBot: Goodbye!")
            break

        bot_response = chatbot_reply(user_input)
        print("ChatBot:", bot_response)
        log_chat(user_input, bot_response)

if __name__ == "__main__":
    main()
