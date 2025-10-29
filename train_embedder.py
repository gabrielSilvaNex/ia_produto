import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.nn import l2_normalize # <-- IMPORTAÇÃO IMPORTANTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pathlib
import random
from tqdm import tqdm

# --- CONFIGURAÇÕES DO METRIC LEARNING ---
CONFIG = {
    "DATASET_DIR": "todos_os_produtos",
    "MODEL_OUTPUT_PATH": "models/product_embedder.keras",
    "IMG_HEIGHT": 300,
    "IMG_WIDTH": 300,
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "LEARNING_RATE": 1e-4,
    "MARGIN": 0.5
}
# ---------------------

def create_triplets(image_paths, image_labels):
    print("Gerando triplets (apenas caminhos)...")
    label_to_paths = {}
    for path, label in zip(image_paths, image_labels):
        if label not in label_to_paths:
            label_to_paths[label] = []
        label_to_paths[label].append(path)

    anchor_paths, positive_paths, negative_paths = [], [], []
    all_labels = list(label_to_paths.keys())
    num_triplets_to_generate = len(image_paths) * 50

    for _ in tqdm(range(num_triplets_to_generate), desc="Criando triplets"):
        anchor_path = random.choice(image_paths)
        anchor_label = image_labels[image_paths.index(anchor_path)]
        if len(label_to_paths[anchor_label]) < 2: continue
        positive_path = random.choice(label_to_paths[anchor_label])
        while positive_path == anchor_path:
            positive_path = random.choice(label_to_paths[anchor_label])
        negative_label = random.choice(all_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(all_labels)
        negative_path = random.choice(label_to_paths[negative_label])
        anchor_paths.append(anchor_path)
        positive_paths.append(positive_path)
        negative_paths.append(negative_path)

    print(f"Gerados {len(anchor_paths)} triplets.")
    return anchor_paths, positive_paths, negative_paths

def load_and_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]])
    return image

def load_and_preprocess_triplet(anchor_path, positive_path, negative_path):
    return (
        load_and_preprocess(anchor_path),
        load_and_preprocess(positive_path),
        load_and_preprocess(negative_path)
    ), 0

def create_embedding_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3),
        include_top=False, weights='imagenet'
    )
    base_model.trainable = True
    print("Descongelando as últimas 20 camadas do EfficientNetV2B0...")
    for layer in base_model.layers[:-20]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
             layer.trainable = False

    inputs = tf.keras.layers.Input(shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Usamos a função 'l2_normalize' importada diretamente
    outputs = tf.keras.layers.Lambda(
        lambda t: l2_normalize(t, axis=-1), 
        name="l2_normalization",
        output_shape=(1280,) # Necessário para salvar/carregar
    )(x)

    return tf.keras.Model(inputs, outputs, name="embedding_model")

def TripletLoss(y_true, y_pred, margin=CONFIG["MARGIN"]):
    anchor_emb, positive_emb, negative_emb = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
    loss = pos_dist - neg_dist + margin
    loss = tf.maximum(0.0, loss)
    return tf.reduce_mean(loss)

def main():
    print("--- Treinamento com Metric Learning (Triplet Loss Manual) ---")
    data_dir = pathlib.Path(CONFIG["DATASET_DIR"])

    print("Procurando por imagens JPG, JPEG e PNG...")
    data_dir = pathlib.Path(CONFIG["DATASET_DIR"])

    # Cria uma lista de padrões de glob
    patterns = ['*/*.jpg', '*/*.jpeg', '*/*.png', '*/*.bmp']

    # Itera sobre os padrões e junta os resultados
    image_paths = []
    for pattern in patterns:
        image_paths.extend(data_dir.glob(pattern))

    # Converte todos os caminhos para string
    image_paths = [str(p) for p in image_paths]

    print(f"Encontrados {len(image_paths)} arquivos de imagem.")
    if not image_paths:
        print(f"ERRO: Nenhuma imagem encontrada."); return

    image_labels_text = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    le = LabelEncoder()
    image_labels_numeric = le.fit_transform(image_labels_text)
    print(f"Encontradas {len(image_paths)} imagens de {len(le.classes_)} produtos.")

    anchor_paths, positive_paths, negative_paths = create_triplets(image_paths, image_labels_numeric)
    (train_a, val_a, train_p, val_p, train_n, val_n) = train_test_split(anchor_paths, positive_paths, negative_paths, test_size=0.2, random_state=42)

    print("Criando datasets (modo serial para evitar crash no macOS)...")
    train_ds = tf.data.Dataset.from_tensor_slices((train_a, train_p, train_n))
    train_ds = train_ds.shuffle(len(train_a))
    train_ds = train_ds.map(load_and_preprocess_triplet, num_parallel_calls=None)
    train_ds = train_ds.batch(CONFIG["BATCH_SIZE"])
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_a, val_p, val_n))
    val_ds = val_ds.map(load_and_preprocess_triplet, num_parallel_calls=None) 
    val_ds = val_ds.batch(CONFIG["BATCH_SIZE"])
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    embedder = create_embedding_model()

    anchor_input = tf.keras.layers.Input(shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3), name="anchor_input")
    positive_input = tf.keras.layers.Input(shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3), name="positive_input")
    negative_input = tf.keras.layers.Input(shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3), name="negative_input")

    anchor_emb = embedder(anchor_input)
    positive_emb = embedder(positive_input)
    negative_emb = embedder(negative_input)

    output = tf.keras.layers.Lambda(
        lambda embeddings: tf.stack(embeddings, axis=1),
        name="stacked_embeddings"
    )([anchor_emb, positive_emb, negative_emb])

    train_model = tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=output)

    print("\nIniciando compilação...")
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]), loss=TripletLoss)
    embedder.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7)

    print("\nIniciando Treinamento (Triplet Loss Manual)...")
    train_model.fit(
        train_ds,
        epochs=CONFIG["EPOCHS"],
        validation_data=val_ds,
        callbacks=[early_stopping, reduce_lr]
    )

    print("\nTreinamento concluído. Salvando o modelo de embedding...")
    os.makedirs(os.path.dirname(CONFIG["MODEL_OUTPUT_PATH"]), exist_ok=True)
    embedder.save(CONFIG["MODEL_OUTPUT_PATH"])

    print(f"✅ Modelo de embedding (com Lambda corrigida) salvo em '{CONFIG['MODEL_OUTPUT_PATH']}'!")

if __name__ == "__main__":
    main()