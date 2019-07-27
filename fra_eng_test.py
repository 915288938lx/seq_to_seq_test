from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra.txt'

# Vectorize the data.
input_texts = [] # 输入文本序列
target_texts = [] # 目标文本序列
input_characters = set() # 输入单词集合(不重复)
target_characters = set() # 目标单词集合(不重复)
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n') # 原始数据所有行的集合
for line in lines[: min(num_samples, len(lines) - 1)]: #循环选定的前10000行
    input_text, target_text = line.split('\t') # 将输入文本序列和目标文本序列分开
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text) # 将所有输入文本序列组装进列表input_texts
    target_texts.append(target_text) # 将所有目标文本序列组装进列表target_texts
    for char in input_text: #对于输入序列中的每一个单词
        if char not in input_characters:#如果不在输入单词集合set()中
            input_characters.add(char) #则将单词添加进set()集合中
    for char in target_text:#同理将目标单词无重复的添加到目标单词集合set()中
        if char not in target_characters:#同理将目标单词无重复的添加到目标单词集合set()中
            target_characters.add(char)#同理将目标单词无重复的添加到目标单词集合set()中

input_characters = sorted(list(input_characters)) #将输入单词集合转换成有序列表
target_characters = sorted(list(target_characters)) # 同理将目标单词集合转换成有序列表
num_encoder_tokens = len(input_characters) # 编码令牌长度==不重复的输入单词的数量
num_decoder_tokens = len(target_characters) # 解码令牌长度==不重复的目标单词数量
max_encoder_seq_length = max([len(txt) for txt in input_texts]) #最大编码序列长度
max_decoder_seq_length = max([len(txt) for txt in target_texts]) #最大解码序列长度

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict( #返回各个不重复输入单词的索引[(char1,0),(char2,1),(char3,2),(char4,3).......]
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict( ##返回各个不重复目标单词的索引[(char1,0),(char2,1),(char3,2),(char4,3).......]
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros( #(输入序列组数,输入序列中最长的序列的单词数量, 所有输入序列中不重复单词数量)
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros( #(输入序列组数,目标序列中最长的序列的单词数量, 所有目标序列中不重复单词数量)
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros( #(输入序列组数,目标序列中最长的序列的单词数量, 所有目标序列中不重复单词数量)
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)): #[(0,(input1,target1)),(1,(input2,target2)),(2,(input3,target3)),....]
    for t, char in enumerate(input_text):# [(0,input1),(1,input2),(2,input3)....]
        encoder_input_data[i, t, input_token_index[char]] = 1. # one_hot 编码输入数据
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1. # one_hot 编码输入数据
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1. #

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(1000,1010):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)