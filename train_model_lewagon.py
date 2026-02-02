#===========================================================================================================================
#========================================BLOCK 1: CONFIGURATION & IMPORTS===================================================
#===========================================================================================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
class Config:
    SEED = 42  # Ensuring that everyone get the exact same split!
    BATCH_SIZE = 32
    EPOCHS = 15
    IMAGE_SIZE = (224, 224)
    LR = 1e-3  # Standard learning rate for frozen models
    
    # Kaggle Paths
    BASE_PATH = '/kaggle/input/covidx-cxr2/'
    
    # Text Files
    TRAIN_TXT = os.path.join(BASE_PATH, 'train.txt')
    VAL_TXT   = os.path.join(BASE_PATH, 'val.txt')
    TEST_TXT  = os.path.join(BASE_PATH, 'test.txt')
    
    # Image Folders
    TRAIN_DIR = os.path.join(BASE_PATH, 'train')
    VAL_DIR   = os.path.join(BASE_PATH, 'val')
    TEST_DIR  = os.path.join(BASE_PATH, 'test')

# Set Seed for Reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)
print("Configuration set. Seed fixed to:", Config.SEED)



#===========================================================================================================================
#========================================BLOCK 2: REORGANIZE DATASET========================================================
#===========================================================================================================================

# --- 1. HELPER FUNCTION TO LOAD DATA ---
def read_data(txt_path, image_folder):
    """Reads a text file and adds the full path to the image folder."""
    # Read the file (Space separated)
    df = pd.read_csv(txt_path, sep=' ', header=None)
    df.columns = ['patient_id', 'filename', 'label', 'source']
    
    # Create the full path column so the generator can find the image later
    df['path'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
    return df

# --- 2. LOAD EVERYTHING ---
print("Loading original datasets...")
df_train = read_data(Config.TRAIN_TXT, Config.TRAIN_DIR)
df_val   = read_data(Config.VAL_TXT,   Config.VAL_DIR)
df_test  = read_data(Config.TEST_TXT,  Config.TEST_DIR)

# Combine them all into one big list
full_df = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
print(f"Total images found: {len(full_df)}")

# --- 3. BALANCE THE DATA ---
# Balancing data BEFORE splitting to keep things fair
min_count = full_df['label'].value_counts().min()
print(f"Balancing dataset to {min_count} images per class...")

# Take a random sample of 'min_count' from each class
balanced_df = full_df.groupby('label').apply(
    lambda x: x.sample(min_count, random_state=Config.SEED)
).reset_index(drop=True)

# Shuffle the final dataframe
balanced_df = balanced_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

# --- 4. CREATE NEW SPLITS ---
# Split 1: 80% for Training, 20% for Temp (Val + Test)
train_df, temp_df = train_test_split(
    balanced_df, 
    train_size=0.8, 
    stratify=balanced_df['label'], # Ensures 50/50 split in train
    random_state=Config.SEED
)

# Split 2: Split the Temp (20%) into half for Val and half for Test
val_df, test_df = train_test_split(
    temp_df, 
    train_size=0.5, 
    stratify=temp_df['label'], 
    random_state=Config.SEED
)

print(f"New Train Size: {len(train_df)}")
print(f"New Val Size:   {len(val_df)}")
print(f"New Test Size:  {len(test_df)}")



#===========================================================================================================================
#=============================================BLOCK 3: GENERATORS===========================================================
#===========================================================================================================================

# --- PREPROCESSING ---
# Using the official DenseNet preprocessing function
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,       # Rotate slightly
    horizontal_flip=True,    # Flip left/right
    fill_mode='nearest'
)

# Test/Val data should NOT be augmented, only preprocessed
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# --- FLOWS ---
print("Creating Data Generators...")

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=Config.IMAGE_SIZE,
    batch_size=Config.BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=Config.SEED
)

val_gen = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=Config.IMAGE_SIZE,
    batch_size=Config.BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=Config.IMAGE_SIZE,
    batch_size=Config.BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)



#===========================================================================================================================
#=============================================BLOCK 4: SEQUENTIAL MODEL=====================================================
#===========================================================================================================================

# --- BUILD MODEL ---
# 1. Download the Base Model (DenseNet121)
base_model = DenseNet121(
    include_top=False,    # Remove the original "1000 classes" head
    weights='imagenet',   # Use pre-trained knowledge
    input_shape=(224, 224, 3)
)

# 2. Freeze the Base Model
base_model.trainable = False 

# 3. Build the Sequential Stack
model = Sequential([
    base_model,                                 
    GlobalAveragePooling2D(),                   
    Dense(128, activation='relu'),              
    BatchNormalization(),                       
    Dropout(0.2),                               
    Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LR),
    loss='binary_crossentropy', # Matches Sigmoid
    metrics=['accuracy']
)

print("Model built successfully!")
model.summary()



#===========================================================================================================================
#=================================================BLOCK 5: TRAINING=========================================================
#===========================================================================================================================

# --- CALLBACKS ---
callbacks = [
    # Stop if validation loss doesn't improve for 5 epochs
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    
    # Save the best version of the model
    ModelCheckpoint('best_covid_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    
    # Slow down learning rate if we get stuck
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
]

# --- TRAIN ---
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=Config.EPOCHS,
    callbacks=callbacks
)



#===========================================================================================================================
#=================================================BLOCK 6: EVALUATION=======================================================
#===========================================================================================================================

# --- EVALUATION ---
print("\n--- Final Evaluation on Test Set ---")

# 1. Load the best weights
model.load_weights('best_covid_model.keras')

# 2. Predict
predictions = model.predict(test_gen)

# 3. Convert probabilities to classes (Threshold 0.5)
y_pred = (predictions > 0.5).astype(int)
y_true = test_gen.classes

# 4. Show Results
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix (Merged Dataset)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))



#===========================================================================================================================
#=================================================BLOCK 7: SAVE FINAL MODEL=================================================
#===========================================================================================================================

# --- SAVE MODEL ---
# 1. Save the final state of the model
save_path = 'my_final_covid_model.keras'
print(f"Saving final model to {save_path}...")
model.save(save_path)

print("Model saved successfully!")

# Note for Teammates:
# To load this model later, they just need to run:
# new_model = tf.keras.models.load_model('my_final_covid_model.keras')


