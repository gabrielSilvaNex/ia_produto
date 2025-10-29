import os
import numpy as np
import tensorflow as tf
from tensorflow.nn import l2_normalize 
from tqdm import tqdm
import pathlib
import ssl
import builtins

builtins.l2_normalize = l2_normalize

CONFIG = {
    "CLASSIFIER_MODEL_PATH": "models/product_embedder.keras",
    "REFERENCE_IMAGES_DIR": "sbp_prints",
    "OUTPUT_LIBRARY_PATH": "models/biblioteca_produtos.npz",
    "IMG_HEIGHT": 300,
    "IMG_WIDTH": 300
}

def create_embeddings_library():
    print("--- Gerando Biblioteca de Embeddings ---")
    try:
        # Passamos a função 'l2_normalize' importada, com o nome da camada
        feature_extractor = tf.keras.models.load_model(
            CONFIG['CLASSIFIER_MODEL_PATH'],
            safe_mode=False 
        )
        print("✅ Modelo extrator carregado.")
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao carregar modelo: {e}"); return

    data_dir = pathlib.Path(CONFIG["REFERENCE_IMAGES_DIR"])
    image_paths = [str(p) for p in data_dir.glob('*/*.*')]
    if not image_paths:
        print(f"❌ ERRO CRÍTICO: Nenhuma imagem encontrada."); return

    print(f"Encontradas {len(image_paths)} imagens. Gerando embeddings...")
    all_embeddings, all_product_ids = [], []

    for path in tqdm(image_paths, desc="Gerando embeddings"):
        product_id = os.path.basename(os.path.dirname(path))
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]])
        img_batch = np.expand_dims(img, axis=0)

        # O modelo já retorna o embedding L2 normalizado
        embedding_final = feature_extractor.predict(img_batch, verbose=0)[0]
        all_embeddings.append(embedding_final)
        all_product_ids.append(product_id)

    os.makedirs(os.path.dirname(CONFIG["OUTPUT_LIBRARY_PATH"]), exist_ok=True)
    np.savez_compressed(CONFIG["OUTPUT_LIBRARY_PATH"], embeddings=np.array(all_embeddings), ids=np.array(all_product_ids))
    print("\n--- ✅ Biblioteca de Embeddings criada com sucesso! ---")

if __name__ == "__main__":
    create_embeddings_library()