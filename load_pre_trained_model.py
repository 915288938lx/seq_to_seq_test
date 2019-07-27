from keras.models import load_model
model = load_model('s2s.h5')
# model.summary()
input_seq = input('input what you want to translate and return:')
seq = model.predict(input_seq)
print(seq)
