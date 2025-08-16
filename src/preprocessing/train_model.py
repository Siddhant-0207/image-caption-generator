import os
import pickle
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ===== Import generator & variables from create_sequences =====
from create_sequences import (
    data_generator, cleaned_captions, image_features,
    tokenizer, MAX_LENGTH, VOCAB_SIZE, BATCH_SIZE
)

# ===== Model parameters =====
EMBEDDING_DIM = 256
LSTM_UNITS = 256
LEARNING_RATE = 0.001
EPOCHS = 20

# ===== Image feature extractor branch =====
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(EMBEDDING_DIM, activation='relu')(fe1)

# ===== Text sequence branch =====
inputs2 = Input(shape=(MAX_LENGTH,))
se1 = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(LSTM_UNITS)(se2)

# ===== Decoder (merge) =====
decoder1 = add([fe2, se3])
decoder2 = Dense(LSTM_UNITS, activation='relu')(decoder1)
outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder2)

# ===== Compile model =====
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE))

model.summary()

# ===== Steps per epoch =====
total_samples = sum(len(caps) * (len(caps[0].split()) + 1) for caps in cleaned_captions.values())
steps = total_samples // BATCH_SIZE

# ===== Train =====
print("🚀 Starting training...")
model.fit(
    data_generator(cleaned_captions, image_features, tokenizer, MAX_LENGTH, VOCAB_SIZE, BATCH_SIZE),
    steps_per_epoch=steps,
    epochs=EPOCHS,
    verbose=1
)

# ===== Save model =====
MODEL_SAVE_PATH = os.path.join("src","model", "caption_model.keras")
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
