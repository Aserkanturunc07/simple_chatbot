"""
This module creates a chatbot with simple memory
capabilities and with terminal compatibility
"""

from transformers import pipeline, Conversation

def chatbot():
    # Initializing the conversational pipeline and model
    model = "facebook/blenderbot-400M-distill"
    converse = pipeline("conversational", model = model)

    print("Hello! I am a chatbot. Type 'exit' to end the conversation.")

    # Creating a conversation object
    conversation = Conversation()

    #Looping until user types exit
    while True:
        # Getting user input
        user_input = input("You: ")

        # Checking if the user wants to exit
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Adding user input to the conversation
        conversation.add_user_input(user_input)

        # Getting the model response using the converse object
        response = converse(
            conversation,
            max_length=128, 
            # Maximum length of the sequence to be generated
            # If max length is exceeded, the input is automatically trimmed
            # to contain the last 128 tokens
            num_return_sequences=1, # just one sequence returned, not multiple)
            top_k=50, # The number of highest probability vocabulary tokens to keep
            top_p=0.95, # making sure the response is generated using most probable responses
            do_sample=True)

        # Print model response
        print("Chatbot:", response.generated_responses[-1])

if __name__ == "__main__":
    chatbot() #running the chatbot
