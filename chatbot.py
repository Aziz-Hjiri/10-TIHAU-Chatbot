#Building a chatbot with Deep NLP
import tensorflow1 as tf
import numpy as np 
import re
import time
 # DATA PREPROCESSING
conversations =open("movie_conversations.tsv" , encoding="utf-8", errors= "ignore").read().split('\n')
lines = open("movie_lines.tsv" , encoding="utf-8", errors= "ignore").read().split('\n')
#creating a dictionary for the ids and the lines associated to them
id2line = {}
for line in lines: 
    _line = line.split('\t')
    if len(_line)==5: 
        id2line[_line[0]]=_line[4]
        
#Creating a list of all the conversations
conversations_ids = []
for conversation in conversations[:-1]: 
    c = conversation.split('\t')[-1][1:-1].replace("'","")
    conversations_ids.append(c.split(" "))
#Getting separately the Q&A
questions=[]
answers=[]
for cnv in conversations_ids:
    for i in range(len(cnv)-1):
            if cnv[i] in id2line:
                questions.append(id2line[cnv[i]])
            if cnv[i+1] in id2line:

                answers.append(id2line[cnv[i+1]])
#Cleaning function: 
def clean_txt(text):
    text = text.lower()
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()#""'',+=.?;:*]", "", text)
    return text
#Cleaning the questions
clean_questions = []
for question in questions: 
    clean_questions.append(clean_txt(question))
#Cleaning the answers
clean_answers = []
for answer in answers: 
    clean_answers.append(clean_txt(answer))
#Map each word with its number of occurences
w2c={}
for qst in clean_questions:
    for word in qst.split():
        if word not in w2c:
            w2c[word] =1
        else:
            w2c[word] +=1
for answer in clean_answers:
    for word in answer.split():
        if word not in w2c:
            w2c[word] =1
        else:
            w2c[word] +=1
#Tokenization and filtering unfrequent words:
threshhold = 20
qw2int = {}
word_number=0
for word , count in w2c.items(): 
    if count >= threshhold:
        qw2int[word]= word_number
        word_number +=1
aw2int = {}
word_number=0
for word , count in w2c.items(): 
    if count >= threshhold:
        aw2int[word]= word_number
        word_number +=1
        
tokens = ["<PAD>", "<EOS>","<OUT>" ,"<SOS>"]
for token in tokens:
    qw2int[token]= len(qw2int)+1
for token in tokens:
    aw2int[token]= len(aw2int)+1
# inverse Dictionnary of words (integer to words)
aint2word={w_i: w for w, w_i in aw2int.items() }

#Adding EOS
for i in range(len(clean_answers)):
    clean_answers[i]+= " <EOS>"
quest2int = []
for qst in clean_questions: 
    associated_int=[]
    for word in qst.split():
        if word not in qw2int:
            associated_int.append(qw2int["<OUT>"])
        else:
            associated_int.append(qw2int[word])
    quest2int.append(associated_int)
answer2int = []
for answer in clean_answers: 
    associated_int=[]
    for word in answer.split():
        if word not in aw2int:
            associated_int.append(aw2int["<OUT>"])
        else:
            associated_int.append(aw2int[word])
    answer2int.append(associated_int)

#Sorting Q&A based on their length
qa_pairs = [(q, a) for q, a in zip(quest2int, answer2int) if a is not None]

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 26):
    for i, qa_pair in enumerate(qa_pairs):
        question, answer = qa_pair
        if len(question) == length: 
            sorted_clean_questions.append(question)
            sorted_clean_answers.append(answer)
        
####### Building the Model (Seq2seq)########
#Creating placeholders
# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 
# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
 
# Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words,
                  questions_num_words, encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 
 
 
########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########
 
 
 
# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(aw2int),
                                                       len(qw2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       qw2int)
 
# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 

    
    

