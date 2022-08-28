import openai

def getOAI(text):
    dialogue = "The following is a conversation between a mental health chatbot named Keagle and a student named User \nUser: " + text + " \nOlive: "
    openai.api_key = "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"
    response = openai.Completion.create(
        engine = "text-davinci-002",
        prompt = dialogue,
            temperature = 0.7,
            max_tokens = 2048
    )
    return response.choices[0].text