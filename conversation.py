from transformers import pipeline, Conversation


def chatbot():
    # Initialize the conversational pipeline
    converse = pipeline("conversational",model = "facebook/blenderbot-400M-distill" )

    print("Hello! I am a chatbot. Type 'exit' to end the conversation.")

    # Create a conversation object
    conversation = Conversation()

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Add user input to the conversation
        conversation.add_user_input(user_input)

        # Get model response
        response = converse(
            conversation,
            max_length=500,  # Maximum length of the sequence to be generated
            num_return_sequences=1,  # Number of sequences to be returned
            top_k=50,  # The number of highest probability vocabulary tokens to keep for top-k-filtering
            top_p=0.95,  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation
            do_sample=True,)

        # Print model response
        print("Chatbot:", response.generated_responses[-1])


if __name__ == "__main__":
    chatbot()
