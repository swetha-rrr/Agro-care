import chainlit as cl  # Getting chainlit
from langchain_core.prompts import ChatPromptTemplate  # For prompts
from langchain_groq import ChatGroq  # Inferencing
from langchain_core.output_parsers import StrOutputParser  # Parsers
from dotenv import load_dotenv  # For detecting env
import os

# Load environment variables
load_dotenv()

# Set the GROQ API key from the environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Getting the API key

# Define the prompt template
prompt = """
    (system: You are a crop assistant, answer the user queries related to specific topic)
    (user: Question: {question})  
"""

# Create a prompt instance
promptinstance = ChatPromptTemplate.from_template(prompt)

# Define the assistant function to handle incoming messages
@cl.on_message  # Decorator for incoming messages
async def assistant(message: cl.Message):
    input_text = message.content  # Get the content of the incoming message
    print(f"Received question: {input_text}")  # Log the incoming question

    # Initialize the language model
    groqllm = ChatGroq(model="llama3-8b-8192", temperature=0)  # llama3 model 
    output = StrOutputParser() 
    chain = promptinstance | groqllm | output  # Create the processing chain
    
    try:
        # Invoke the chain with the input text
        res = chain.invoke({'question': input_text})  
        print(f"Response generated: {res}")  # Log the response
        
        # Send the response back to the user
        await cl.Message(content=res).send()  
    except Exception as e:
        # Log any errors that occur
        print(f"Error generating response: {str(e)}")  
        
        # Send an error message back to the user
        await cl.Message(content="Error processing your request.").send()  

if __name__ == "__main__":
    # Run the Chainlit app
    cl.run(assistant, port=8000)  # Ensure this matches the port in app.py
