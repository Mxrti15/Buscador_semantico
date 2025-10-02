from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
import pandas as pd


app = Flask(__name__)

# 1. Cargar modelo de embeddings(embeddings + FAQs)
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Base de conocimiento (FAQ del restaurante)
df = pd.read_csv("faqs.csv")    # lee el archivo
faq_questions = df["pregunta"].tolist() # horario, menu, etc..
faq_answers = df["respuesta"].tolist()  # nuestro horario...

# 3. Precalcular embeddings de las FAQs
faq_embeddings = emb_model.encode(faq_questions) # codifica la entrada de texto

# Cargar modelo generativo ligero (Flan-T5 small)(LLM)
gen_model = pipeline("text2text-generation", model="google/flan-t5-small")

@app.route("/")
def index():
    return render_template("index.html")    # redireccion a la vista

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"].strip()
    user_input_clean = user_input.lower()  # normalizamos

    # --- Respuestas directas para saludos y small talk ---
    if any(word in user_input_clean for word in ["hola", "buenas", "hey"]):
        return jsonify({"response": "¡Hola! Bienvenido al Restaurante Girona 🍴. ¿Quieres saber nuestros horarios o el menú?"})
    if "como estas" in user_input_clean or "qué tal" in user_input_clean:
        return jsonify({"response": "¡Muy bien! Gracias por preguntar 😊. ¿Quieres que te ayude con el menú o las reservas?"})

    # 4. Convertir input del usuario en embedding
    user_embedding = emb_model.encode([user_input_clean])

    # 5. Calcular similitud con cada FAQ
    similarities = cosine_similarity(user_embedding, faq_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    # 6. Umbral de confianza
    if best_score > 0.6:
        answer = faq_answers[best_match_idx]
    else:
        # --- Fallback directo ---
        fallbacks = [
            "Lo siento, solo puedo ayudarte con información sobre el Restaurante Girona (horarios, menú, reservas, ubicación).",
            "Esa es una buena pregunta 😅, pero yo solo sé cosas del Restaurante Girona.",
            "Solo puedo darte información sobre nuestro restaurante en Girona 🍴."
        ]
        import random
        answer = random.choice(fallbacks)

    return jsonify({"response": answer})



if __name__ == "__main__":
    app.run(debug=True)
