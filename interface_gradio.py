import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os

# ─────────────────────────────────────────
# 1. Chargement des modèles
# ─────────────────────────────────────────
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE   = (224, 224)

# Icônes emoji par catégorie
ICONS = {
    'cardboard': '📦',
    'glass':     '🍶',
    'metal':     '🥫',
    'paper':     '📄',
    'plastic':   '🧴',
    'trash':     '🗑️',
}

# Couleurs par catégorie
COLORS = {
    'cardboard': '#C8872A',
    'glass':     '#4DB6AC',
    'metal':     '#78909C',
    'paper':     '#5C9BD6',
    'plastic':   '#AB47BC',
    'trash':     '#EF5350',
}

def load_models():
    models = {}
    model_files = {
        'CNN Simple':      'model_cnn_simple.h5',
        'CNN Profond':     'model_cnn_profond.h5',
        'Transfer Learning (MobileNetV2)': 'model_transfer_learning.h5',
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
                print(f"✅ {name} chargé")
            except Exception as e:
                print(f"⚠️  {name} — erreur : {e}")
        else:
            print(f"⚠️  {name} introuvable : {path}")
    return models

MODELS = load_models()

# ─────────────────────────────────────────
# 2. Fonction de prédiction
# ─────────────────────────────────────────
def predict(image):
    if image is None:
        return build_empty_html(), "{}"

    img = Image.fromarray(image).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    results = {}
    for model_name, model in MODELS.items():
        preds       = model.predict(arr, verbose=0)[0]
        idx         = int(np.argmax(preds))
        label       = CATEGORIES[idx]
        confidence  = float(preds[idx])
        all_probs   = {CATEGORIES[i]: float(preds[i]) for i in range(len(CATEGORIES))}
        results[model_name] = {
            'label':      label,
            'confidence': confidence,
            'probs':      all_probs,
        }

    html = build_results_html(results)
    raw  = {k: {'prediction': v['label'], 'confidence': round(v['confidence'], 3)}
            for k, v in results.items()}
    return html, json.dumps(raw, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────
# 3. Génération du HTML de résultats
# ─────────────────────────────────────────
def build_empty_html():
    return """
    <div style="
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        padding: 60px 20px;
        color: #888;
        background: #1a1a2e;
        border-radius: 16px;
    ">
        <div style="font-size: 48px; margin-bottom: 12px;">🖼️</div>
        <p style="font-size: 16px;">Uploadez une image pour voir les prédictions</p>
    </div>
    """

def conf_bar(value, color):
    pct = round(value * 100, 1)
    return f"""
    <div style="margin: 4px 0;">
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#aaa; margin-bottom:3px;">
        </div>
        <div style="background:#2d2d4e; border-radius:999px; height:8px; overflow:hidden;">
            <div style="
                width:{pct}%;
                height:100%;
                background: linear-gradient(90deg, {color}aa, {color});
                border-radius:999px;
                transition: width 0.8s ease;
            "></div>
        </div>
    </div>
    """

def build_results_html(results):
    # Trouver le consensus (catégorie la plus votée)
    votes = {}
    for r in results.values():
        lbl = r['label']
        votes[lbl] = votes.get(lbl, 0) + 1
    winner = max(votes, key=votes.get)
    winner_icon  = ICONS.get(winner, '♻️')
    winner_color = COLORS.get(winner, '#4CAF50')

    cards_html = ""
    model_short = {
        'CNN Simple':      ('CNN Simple',    '🔵', '#4169E1'),
        'CNN Profond':     ('CNN Profond',   '🟢', '#2E8B57'),
        'Transfer Learning (MobileNetV2)': ('Transfer Learning', '🔴', '#DC143C'),
    }

    for i, (model_name, data) in enumerate(results.items()):
        label      = data['label']
        conf       = data['confidence']
        probs      = data['probs']
        color      = COLORS.get(label, '#888')
        icon       = ICONS.get(label, '♻️')
        short, _, mcolor = model_short.get(model_name, (model_name, '⚪', '#888'))
        conf_pct   = round(conf * 100, 1)

        # Barres des 3 meilleures catégories
        top3       = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        bars_html  = ""
        for cat, prob in top3:
            cat_color = COLORS.get(cat, '#888')
            cat_icon  = ICONS.get(cat, '')
            bars_html += f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                    <span style="font-size:12px; color:#ccc;">{cat_icon} {cat}</span>
                    <span style="font-size:12px; font-weight:700; color:{cat_color};">{round(prob*100,1)}%</span>
                </div>
                <div style="background:#2d2d4e; border-radius:999px; height:7px; overflow:hidden;">
                    <div style="width:{round(prob*100,1)}%; height:100%;
                        background:linear-gradient(90deg,{cat_color}66,{cat_color});
                        border-radius:999px;"></div>
                </div>
            </div>
            """

        is_winner_model = (label == winner)
        border = f"2px solid {color}" if is_winner_model else "1px solid #2d2d4e"

        cards_html += f"""
        <div style="
            background: #16213e;
            border: {border};
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            animation: fadeIn 0.4s ease {i*0.15}s both;
        ">
            <!-- En-tête modèle -->
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:10px; height:10px; border-radius:50%; background:{mcolor};"></div>
                    <span style="font-size:13px; font-weight:600; color:#e0e0e0; letter-spacing:0.3px;">{short}</span>
                </div>
                {"<span style='font-size:10px; background:#ffffff15; color:#ccc; padding:2px 8px; border-radius:999px;'>✓ consensus</span>" if is_winner_model else ""}
            </div>

            <!-- Résultat principal -->
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:16px;">
                <div style="
                    width:52px; height:52px;
                    background: {color}22;
                    border: 1.5px solid {color}55;
                    border-radius:12px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:26px;
                ">{icon}</div>
                <div>
                    <div style="font-size:20px; font-weight:800; color:{color}; text-transform:capitalize;">{label}</div>
                    <div style="font-size:13px; color:#aaa; margin-top:2px;">Confiance : <strong style="color:#fff;">{conf_pct}%</strong></div>
                </div>
            </div>

            <!-- Top 3 barres -->
            <div>{bars_html}</div>
        </div>
        """

    html = f"""
    <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 {winner_color}44; }}
            50%       {{ box-shadow: 0 0 0 10px transparent; }}
        }}
    </style>
    <div style="font-family: 'Segoe UI', sans-serif; background: #0f0f23; border-radius: 18px; padding: 24px;">

        <!-- Bandeau consensus -->
        <div style="
            background: linear-gradient(135deg, {winner_color}22, {winner_color}0a);
            border: 1.5px solid {winner_color}55;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 22px;
            animation: pulse 2s infinite;
            display: flex; align-items: center; gap: 18px;
        ">
            <div style="font-size: 48px;">{winner_icon}</div>
            <div>
                <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;">Résultat consensus</div>
                <div style="font-size: 28px; font-weight: 900; color: {winner_color}; text-transform: capitalize;">{winner}</div>
                <div style="font-size: 12px; color: #aaa; margin-top: 4px;">{votes[winner]}/{len(results)} modèles d'accord</div>
            </div>
        </div>

        <!-- Titre section -->
        <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px;">
            Détail par modèle
        </div>

        <!-- Cartes modèles -->
        {cards_html}
    </div>
    """
    return html


# ─────────────────────────────────────────
# 4. Interface Gradio
# ─────────────────────────────────────────
CSS = """
body { background: #0a0a1a !important; }

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', sans-serif !important;
}

.title-block {
    text-align: center;
    padding: 32px 0 20px 0;
}

footer { display: none !important; }

/* Upload zone */
.upload-btn {
    border: 2px dashed #4169E1 !important;
    border-radius: 16px !important;
    background: #0f0f23 !important;
}

/* Boutons */
button.primary {
    background: linear-gradient(135deg, #4169E1, #6a3de8) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 12px 28px !important;
}

button.secondary {
    background: #1a1a2e !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    color: #aaa !important;
}
"""

TITLE_HTML = """
<div style="
    text-align: center;
    padding: 28px 0 10px;
    font-family: 'Segoe UI', sans-serif;
">
    <div style="font-size: 36px; margin-bottom: 8px;">♻️</div>
    <h1 style="
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(135deg, #4169E1, #2E8B57, #DC143C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    ">TrashNet — Classificateur de Déchets</h1>
    <p style="color: #888; font-size: 14px; margin: 0;">
        3 modèles CNN analysent votre image et votent pour la catégorie la plus probable
    </p>
    <div style="display:flex; justify-content:center; gap:12px; margin-top:14px; flex-wrap:wrap;">
        <span style="background:#4169E122; border:1px solid #4169E155; color:#7a9ef0;
            padding:4px 12px; border-radius:999px; font-size:11px;">🔵 CNN Simple</span>
        <span style="background:#2E8B5722; border:1px solid #2E8B5755; color:#6dbf9a;
            padding:4px 12px; border-radius:999px; font-size:11px;">🟢 CNN Profond</span>
        <span style="background:#DC143C22; border:1px solid #DC143C55; color:#f07090;
            padding:4px 12px; border-radius:999px; font-size:11px;">🔴 Transfer Learning</span>
    </div>
</div>
"""

CATEGORIES_HTML = """
<div style="
    font-family: 'Segoe UI', sans-serif;
    background: #0f0f23;
    border: 1px solid #1e1e3e;
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 4px;
">
    <div style="font-size:11px; color:#666; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">
        Catégories reconnues
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
        <span style="background:#C8872A22;border:1px solid #C8872A55;color:#d4a060;padding:5px 12px;border-radius:8px;font-size:13px;">📦 Cardboard</span>
        <span style="background:#4DB6AC22;border:1px solid #4DB6AC55;color:#6dcfc8;padding:5px 12px;border-radius:8px;font-size:13px;">🍶 Glass</span>
        <span style="background:#78909C22;border:1px solid #78909C55;color:#9ab0ba;padding:5px 12px;border-radius:8px;font-size:13px;">🥫 Metal</span>
        <span style="background:#5C9BD622;border:1px solid #5C9BD655;color:#80b8e8;padding:5px 12px;border-radius:8px;font-size:13px;">📄 Paper</span>
        <span style="background:#AB47BC22;border:1px solid #AB47BC55;color:#c97fd4;padding:5px 12px;border-radius:8px;font-size:13px;">🧴 Plastic</span>
        <span style="background:#EF535022;border:1px solid #EF535055;color:#f07878;padding:5px 12px;border-radius:8px;font-size:13px;">🗑️ Trash</span>
    </div>
</div>
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0a0a1a",
    block_background_fill="#0f0f23",
    block_border_color="#1e1e3e",
    input_background_fill="#16213e",
)) as demo:

    gr.HTML(TITLE_HTML)

    with gr.Row(equal_height=False):
        # ── Colonne gauche : input ──
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Image à analyser",
                type="numpy",
                height=300,
                elem_classes=["upload-btn"],
            )
            gr.HTML(CATEGORIES_HTML)

            with gr.Row():
                clear_btn  = gr.ClearButton([image_input], value="🗑️ Effacer")
                submit_btn = gr.Button("🔍 Analyser", variant="primary")

        # ── Colonne droite : résultats ──
        with gr.Column(scale=1):
            results_html = gr.HTML(
                value="""
                <div style="
                    font-family:'Segoe UI',sans-serif;
                    text-align:center;
                    padding:80px 20px;
                    color:#444;
                    background:#0f0f23;
                    border-radius:16px;
                    border:1px dashed #1e1e3e;
                ">
                    <div style="font-size:44px;margin-bottom:12px;">♻️</div>
                    <p style="font-size:15px;">Les résultats apparaîtront ici</p>
                    <p style="font-size:12px;color:#333;">3 modèles analyseront votre image</p>
                </div>
                """,
                label="Résultats"
            )
            with gr.Accordion("📋 JSON brut", open=False):
                json_output = gr.Code(language="json", label="")

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[results_html, json_output],
    )
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[results_html, json_output],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
