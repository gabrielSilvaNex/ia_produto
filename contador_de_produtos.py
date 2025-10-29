import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.nn import l2_normalize # <-- 1. IMPORTAÇÃO
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from collections import Counter
import argparse
import builtins # <-- 1. IMPORTAÇÃO

# --- 2. INJEÇÃO DA FUNÇÃO NO ESCOPO GLOBAL ---
# Hack inteligente para consertar a Lambda sem re-treino
builtins.l2_normalize = l2_normalize
# ----------------------------------------------

CONFIG = {
    "AREA_DETECTOR_PATH": "models/area_detector.pt",
    "PRODUCT_DETECTOR_PATH": "models/product_detector.pt",
    "FEATURE_EXTRACTOR_PATH": "models/product_embedder.keras",  
    "EMBEDDINGS_LIBRARY_PATH": "models/biblioteca_produtos.npz",
    "AREA_CONF_THRESHOLD": 0.5,
    "PRODUCT_CONF_THRESHOLD": 0.2,
    "MATCH_SIMILARITY_THRESHOLD": 0.4, # Ajuste este valor se os resultados forem ruins
    "IMG_HEIGHT": 300, "IMG_WIDTH": 300
}

def load_models_and_library():
    print("Carregando modelos e biblioteca...")
    try:
        area_detector = YOLO(CONFIG["AREA_DETECTOR_PATH"])
        product_detector = YOLO(CONFIG["PRODUCT_DETECTOR_PATH"])
        
        # --- 3. CARREGAMENTO DO MODELO CORRIGIDO ---
        # O hack do 'builtins' remove a necessidade de 'custom_objects'
        feature_extractor = tf.keras.models.load_model(
            CONFIG["FEATURE_EXTRACTOR_PATH"],
            safe_mode=False # Ainda necessário para permitir a camada Lambda
        )

        library_data = np.load(CONFIG["EMBEDDINGS_LIBRARY_PATH"], allow_pickle=True)
        library_embeddings = library_data['embeddings']
        library_product_ids = library_data['ids']

        print("✅ Modelos e biblioteca carregados com sucesso!")
        return area_detector, product_detector, feature_extractor, library_embeddings, library_product_ids
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return None, None, None, None, None

def preprocess_for_embedding(image):

    img_resized = tf.image.resize(image, [CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]])
    return np.expand_dims(img_resized, axis=0)

def find_best_match(embedding, library_embeddings, library_product_ids):
    best_match_id, best_similarity = None, -1
    for i, lib_embedding in enumerate(library_embeddings):
        similarity = 1 - cosine(embedding, lib_embedding)
        if similarity > best_similarity:
            best_similarity, best_match_id = similarity, library_product_ids[i]
    return best_match_id, best_similarity

def main(image_path):
    models = load_models_and_library()
    if not all(m is not None for m in models): return
    area_detector, product_detector, feature_extractor, lib_embeds, lib_ids = models

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERRO: Não foi possível ler a imagem em '{image_path}'"); return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results_area = area_detector.predict(img_rgb, conf=CONFIG["AREA_CONF_THRESHOLD"], verbose=False)
    product_counts = Counter()

    for area_result in results_area:
        for box in area_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cropped_area_rgb = img_rgb[y1:y2, x1:x2]
            results_products = product_detector.predict(cropped_area_rgb, conf=CONFIG["PRODUCT_CONF_THRESHOLD"], verbose=False)

            for product_result in results_products:
                for prod_box in product_result.boxes:
                    px1, py1, px2, py2 = map(int, prod_box.xyxy[0])
                    product_image = cropped_area_rgb[py1:py2, px1:px2]
                    if product_image.size == 0: continue

                    preprocessed_product = preprocess_for_embedding(product_image)
                    
                    # O modelo já retorna o embedding normalizado.
                    embedding_final = feature_extractor.predict(preprocessed_product, verbose=False)[0]
                    
                    product_id, similarity = find_best_match(embedding_final, lib_embeds, lib_ids)

                    abs_px1, abs_py1 = x1 + px1, y1 + py1; abs_px2, abs_py2 = x1 + px2, y1 + py2
                    
                    if similarity >= CONFIG["MATCH_SIMILARITY_THRESHOLD"]:
                        product_counts[product_id] += 1
                        label = f"{product_id} ({similarity:.2f})"; color = (0, 255, 0)
                    else:
                        label = f"Desconhecido ({similarity:.2f})"; color = (0, 0, 255)

                    cv2.rectangle(img_bgr, (abs_px1, abs_py1), (abs_px2, abs_py2), color, 2)
                    cv2.putText(img_bgr, label, (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    print("\n--- CONTAGEM FINAL ---")
    if not product_counts:
        print("Nenhum produto foi identificado com confiança.")
    else:
        for product, count in sorted(product_counts.items()):
            print(f"- {product}: {count} unidades")

    output_path = "resultado.jpg"
    cv2.imwrite(output_path, img_bgr)
    print(f"✅ Imagem com resultados salva em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contador de produtos em imagens de gôndola.")
    parser.add_argument("--imagem", type=str, required=True, help="Caminho para a imagem de entrada.")
    args = parser.parse_args()
    main(args.imagem)