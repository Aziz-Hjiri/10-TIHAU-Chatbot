 10 Things I Hate About You DNLP project
This is a project on creating a chatbot with Deep NLP using the seq2seq model. The project is based on the 1999 movie "10 Things I Hate About You". The chatbot is designed to answer questions related to the movie.

Dataset
The dataset for this project is a collection of dialogues from the movie "10 Things I Hate About You". The dataset contains the following files:

movie_lines.tsv: contains the text of each individual line of dialogue
movie_conversations.tsv: contains the sequence of conversation between characters
The dataset was preprocessed by cleaning and transforming it to the format required for the seq2seq model.

Libraries and tools
The following libraries were used in this project:

numpy
tensorflow
re
time
Files
The project consists of the following files:

chatbot.py: the python file the code for the chatbot
movie_lines.tsv: the text file containing the movie dialogues
movie_conversations.tsv: the text file containing the conversation sequences
Running the code
To run the code, you can simply open the chatbot.ipynb file in Jupyter Notebook and run the cells in order.

Model architecture
The seq2seq model is used to build the chatbot. The model consists of an encoder and a decoder. The encoder takes the input sequence of words and outputs a fixed-length vector representation of the sequence. The decoder then takes this vector representation and generates the output sequence of words. The model is trained using the teacher forcing technique.

Output
The chatbot is designed to answer questions related to the movie "10 Things I Hate About You". The user can input any question related to the movie and the chatbot will generate an appropriate answer.
