# WPI-Gemini-Chatbot
A simple application demonstrating integration of Gemini in developing a chatbot for answering Worcester Polytechnic Institute's queries

To run the application, make sure you have streamlit installed. Run the application using command streamlit and run the code chatbot.py using streamlit.

Before that run the data_scraping.py and ih_data_extraction.py to extract the data and store the vector embeddings into pinecone. Setup your own pinecone database and get the API keys and store it in env file.

This project collects data from WPI's public websites and then splits them into chunks and converts into vector embeddings and stores into pinecone. The user then asks query from the frontend, the query is converted to vector and similar documents are retrieved from pinecone based on cosine similarity measure, which is then fed as a context along with the query to Gemini model and the response is generated to the user.

