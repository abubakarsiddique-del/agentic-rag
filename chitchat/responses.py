"""Rotating, concise responses for non-document conversational turns."""

from __future__ import annotations

import random
import threading
from collections import OrderedDict


RESPONSES: dict[str, tuple[str, ...]] = {
    "greeting": (
        "Hi! What would you like to look into?",
        "Hello! What can I help you with?",
        "Hey! Ready when you are.",
    ),
    "first_time": (
        "This is Agentic RAG. Upload a document and ask questions grounded in its contents.",
        "Agentic RAG answers questions from files you upload. Add a PDF or text file to get started.",
        "You can upload documents here, then ask about their contents.",
    ),
    "capabilities": (
        "I answer questions from your uploaded documents and can point to their sources.",
        "I can search your uploaded files, compare them, and summarize their contents.",
        "Upload PDFs or text files, then ask for details, summaries, or comparisons grounded in them.",
    ),
    "how_are_you": (
        "Doing well, thanks. What would you like to work on?",
        "All good here. What can I help you find in your files?",
        "Ready to help. What would you like to ask?",
    ),
    "thanks": (
        "You’re welcome.",
        "Glad I could help.",
        "Anytime. Let me know what else you’d like to check.",
    ),
    "farewell": (
        "Goodbye! Your chats will be here when you return.",
        "Take care. Come back whenever you need to check your files.",
        "See you next time.",
    ),
    "acknowledgment": (
        "Got it.",
        "Understood.",
        "Sounds good.",
    ),
    "user_apology": (
        "No worries.",
        "That’s okay. We can keep going.",
        "No need to apologize.",
    ),
    "compliment": (
        "Thanks. I’m glad the answer was useful.",
        "I appreciate that. Happy to help with the documents.",
        "Thanks for saying so.",
    ),
    "how_to_use": (
        "Upload a PDF or text file, then ask a question about its contents.",
        "Add a document with the upload control, wait until it’s ready, and ask away.",
        "Start by uploading a file. You can then ask for a summary, a detail, or a comparison.",
    ),
    "confusion": (
        "No problem. Upload a file or ask a question about one that’s already loaded.",
        "We can take it one step at a time. What are you trying to find?",
        "Tell me what you’re stuck on, or ask a direct question about your files.",
    ),
    "no_documents_yet": (
        "I can answer from your uploaded documents. Add a PDF or text file first, then ask again.",
        "There aren’t any ready documents in this chat yet. Upload a file and I can help with it.",
        "I’ll need an uploaded document to answer that. Add a PDF or text file to get started.",
    ),
    "document_just_uploaded": (
        "Your document is ready. Ask a question about it whenever you’re ready.",
        "Upload complete. You can now ask about the document’s details or main points.",
        "The file is ready to search. What would you like to find in it?",
    ),
    "positive_feedback": (
        "Glad that worked.",
        "Great. Let me know if you want to check another detail.",
        "Happy to hear it was useful.",
    ),
    "negative_feedback": (
        "Thanks for the feedback. Try narrowing the question or selecting the relevant file.",
        "Sorry that missed the mark. A more specific question may help me find the right passage.",
        "Understood. You can rephrase the question or choose which documents to search.",
    ),
    "casual_request": (
        "I’m best at helping with your uploaded documents. What would you like to find in them?",
        "I can keep the conversation light, but my main job is answering from your files. What should we look up?",
        "Let’s stick to what I can do well: upload a document and ask me about it.",
    ),
}

_MAX_CONVERSATIONS = 2_000
_recent: OrderedDict[str, dict[str, int]] = OrderedDict()
_lock = threading.Lock()


def choose_response(category: str, conversation_id: str) -> str:
    variants = RESPONSES[category]
    key = str(conversation_id)
    with _lock:
        previous_by_category = _recent.setdefault(key, {})
        previous_index = previous_by_category.get(category)
        available = [index for index in range(len(variants)) if index != previous_index]
        selected = random.choice(available or list(range(len(variants))))
        previous_by_category[category] = selected
        _recent.move_to_end(key)
        while len(_recent) > _MAX_CONVERSATIONS:
            _recent.popitem(last=False)
    return variants[selected]
