import math

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora"
LABELS = ["negativo", "neutro", "positivo"]
EMOJI = {"negativo": "😞", "neutro": "😐", "positivo": "😊"}

# Mesmo prompt de sistema usado no treino (não altere, senão o modelo se comporta diferente)
SYSTEM = ("Você classifica a polaridade de avaliações de clientes em português. "
          "Responda apenas com uma palavra: positivo, neutro ou negativo.")

EXEMPLOS = {
    "Positivo": "Produto excelente, chegou antes do prazo e superou minhas expectativas. Recomendo!",
    "Negativo": "Veio com defeito e a loja não respondeu meus contatos. Péssima experiência, não compro mais.",
    "Neutro": "O produto é razoável. Funciona, mas não é nada demais, e a entrega demorou um pouco.",
}


@st.cache_resource(show_spinner="Carregando o modelo (só na primeira vez)...")
def load():
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)  # metade da memória do float32
    model = PeftModel.from_pretrained(base, ADAPTER)
    model = model.merge_and_unload()  # junta o LoRA ao modelo base: inferência mais rápida na CPU
    model.eval()
    return tok, model


@torch.no_grad()
def classify(tok, model, text):
    """Probabilidade de cada classe = log-prob da sequência 'rótulo + <eos>' dado o prompt,
    normalizada entre as 3 classes (softmax). Não depende do primeiro token ser diferente."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids

    scores = []
    for lab in LABELS:
        ans_ids = tok(lab + tok.eos_token, add_special_tokens=False).input_ids
        ids = torch.tensor([prompt_ids + ans_ids])
        logits = model(input_ids=ids).logits[0]
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)
        targets = ids[0, 1:]
        start = len(prompt_ids) - 1
        token_lp = logp[start:].gather(1, targets[start:].unsqueeze(1)).sum().item()
        scores.append(token_lp)

    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return {lab: e / total for lab, e in zip(LABELS, exps)}


st.set_page_config(page_title="Polaridade de avaliações PT-BR", page_icon="🛒")
st.title("🛒 Polaridade de avaliações em PT-BR")
st.caption(
    "Qwen2.5-0.5B-Instruct ajustado com LoRA em avaliações de e-commerce (B2W). "
    "Classifica como negativo, neutro ou positivo."
)

if "texto" not in st.session_state:
    st.session_state.texto = ""

st.write("**Exemplos prontos:**")
cols = st.columns(len(EXEMPLOS))
for col, (nome, txt) in zip(cols, EXEMPLOS.items()):
    if col.button(nome, use_container_width=True):
        st.session_state.texto = txt

texto = st.text_area("Cole uma avaliação:", key="texto", height=140, max_chars=600)

if st.button("Classificar", type="primary"):
    if not texto.strip():
        st.warning("Escreva ou escolha uma avaliação primeiro.")
    else:
        tok, model = load()
        with st.spinner("Classificando..."):
            probs = classify(tok, model, texto.strip())
        pred = max(probs, key=probs.get)
        st.subheader(f"{EMOJI[pred]} {pred.capitalize()}")
        for lab in LABELS:
            st.progress(probs[lab], text=f"{lab}: {probs[lab]:.1%}")

st.divider()
st.markdown(
    f"[Modelo no Hugging Face](https://huggingface.co/{ADAPTER}) · "
    "[Código no GitHub](https://github.com/NS-0111/finetune-llm-polaridade-ptbr)"
)
st.caption(
    "Acurácia de 76,5% e F1 macro de 0,76 em 600 avaliações de teste. "
    "A classe 'neutro' é a mais difícil. Treinado só com avaliações de e-commerce."
)
