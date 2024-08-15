import nltk

nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import wikipedia

rules = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! What's on your mind?",
    "how are you": "I'm doing well, thanks! How about you?",
    "what's your name": "My name is Chatty, nice to meet you!",
    "quit": "Goodbye! It was nice chatting with you.",
    "bye": "See you later! Have a great day.",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "good morning": "Good morning! How can I start your day?",
    "good afternoon": "Good afternoon! How's your day going?",
    "good evening": "Good evening! How can I help you tonight?",
    "i'm feeling sad": "Sorry to hear that. Would you like to talk about it?",
    "i'm feeling happy": "That's great to hear! What's making you happy today?",
}


def process_input(user_input):
    tokens = word_tokenize(user_input)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


def get_wikipedia_info(topic):
    try:
        summary = wikipedia.summary(topic, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Sorry, I'm not sure what you mean by '{topic}'. Did you mean {', '.join(e.options)}?"
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find any information on '{topic}'."


def respond(user_input):
    user_input = process_input(user_input)
    for rule in rules:
        if rule in user_input:
            return rules[rule]
    if "what is" in user_input:
        topic = user_input.split("what is ")[-1]
        return get_wikipedia_info(topic)

    elif "who is" in user_input:
        topic = user_input.split("who is ")[-1]
        return get_wikipedia_info(topic)

    elif "when is" in user_input:
        topic = user_input.split("when is ")[-1]
        return get_wikipedia_info(topic)

    return "I didn't understand that. Can you please rephrase?"


def chatbot():
    print("Welcome to Chatty! I'm here to help.")
    while True:
        user_input = input("You: ")
        response = respond(user_input)
        print("Chatty: " + response)

        # Check if the user wants to quit
        if user_input.lower() in ["quit", "bye"]:
            break


chatbot()
