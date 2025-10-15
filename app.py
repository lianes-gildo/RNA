import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X = np.array([
    [12, 5, 14],   # atividade normal
    [2, 1, 2],     # suspeito (muito proximo da costa, a noite)
    [15, 10, 10],  # normal
    [1, 0.5, 23],  # suspeito
    [14, 7, 16],   # normal
    [3, 1, 3],     # suspeito
])

# Rotulos: 0 = normal, 1 = suspeito
y = np.array([0, 1, 0, 1, 0, 1])

# Modelo
model = Sequential([
    Dense(8, input_shape=(3,), activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(X, y, epochs=20, verbose=0)


novo_dado = np.array([[2, 0.8, 1]])  # barco lento, muito perto da costa, à noite
previsao = model.predict(novo_dado)

novo_dado2 = np.array([[10, 0.10, 20]])
previsao2 = model.predict(novo_dado2)

print("Probabilidade de atividade suspeita: ", previsao[0][0])
print ("Probabilidade de atividade suspeita: ", previsao2[0][0])