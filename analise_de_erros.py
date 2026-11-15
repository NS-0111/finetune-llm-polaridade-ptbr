# %% [markdown]
# # Análise de erros do modelo de polaridade
# Reconstrói o MESMO teste de 600 avaliações, carrega o adapter publicado no Hugging Face,
# salva a previsão de cada avaliação e investiga ONDE o modelo erra.
# Cada "# %%" é uma célula. Rode em ordem. Funciona com GPU (rápido) ou sem (alguns minutos).

# %% 0. Instalação (descomente no Kaggle/Colab)
# !pip install -q transformers peft pandas scikit-learn
# NO KAGGLE, rode também esta linha (o torchao antigo do Kaggle quebra o peft):
# !pip uninstall -y torchao

# %% 1. Dados: mesmo procedimento do treino, para reconstruir o MESMO teste
import random
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
N_TEST = 600
LABELS = ["negativo", "neutro", "positivo"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora"

path = hf_hub_download(repo_id="vladjr/B2W-Reviews01", filename="B2W-Reviews01.csv", repo_type="dataset")
raw = pd.read_csv(path)
df = raw[["review_text", "overall_rating"]].dropna()
df = df[df.review_text.str.len().between(20, 600)]
df["label"] = df.overall_rating.astype(int).map(
    lambda r: "negativo" if r <= 2 else ("neutro" if r == 3 else "positivo"))

def sample_balanced(frame, n_total):
    per_class = n_total // len(LABELS)
    parts = [frame[frame.label == lab].sample(per_class, random_state=SEED) for lab in LABELS]
    return pd.concat(parts).sample(frac=1, random_state=SEED)

test_df = sample_balanced(df, N_TEST)
test_df = test_df.join(raw[["site_category_lv1"]])      # categoria do produto, para a análise
print(test_df.label.value_counts())

# %% 2. Modelo ajustado, baixado do Hugging Face
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER).to(device).eval()

SYSTEM = ("Você classifica a polaridade de avaliações de clientes em português. "
          "Responda apenas com uma palavra: positivo, neutro ou negativo.")

def build_prompt(text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(output):
    out = output.strip().lower()
    for lab in LABELS:
        if out.startswith(lab):
            return lab
    return "invalido"

@torch.no_grad()
def predict(texts, bs=16):
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(texts), bs):
        batch = [build_prompt(t) for t in texts[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += [parse(x) for x in tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return preds

# %% 3. Previsões no teste (uma linha por avaliação)
test_df["previsao"] = predict(test_df.review_text.tolist())
test_df["correto"] = test_df.label == test_df.previsao
test_df["tamanho"] = test_df.review_text.str.len()
test_df.to_csv("predicoes_teste.csv", index=False)

# Conferência: se o teste foi reconstruído certo, deve dar perto de 0,765 (o resultado já documentado)
print("acurácia:", round(test_df.correto.mean(), 3))

# %% 4. Acurácia por nota de 1 a 5 estrelas (onde estão os erros de verdade?)
linhas = []
for nota, g in test_df.groupby("overall_rating"):
    linhas.append({"nota": int(nota), "n": len(g), "acerto": round(g.correto.mean(), 3),
                   **{f"prev_{l}": round((g.previsao == l).mean(), 3) for l in LABELS}})
por_nota = pd.DataFrame(linhas)
print(por_nota.to_string(index=False))

# %% 5. Tipos de erro: vizinhos (envolvem o neutro) ou graves (negativo <-> positivo)?
erros = test_df[~test_df.correto]
graves = erros[((erros.label == "negativo") & (erros.previsao == "positivo")) |
               ((erros.label == "positivo") & (erros.previsao == "negativo"))]
invalidos = erros[erros.previsao == "invalido"]
vizinhos = erros[~erros.index.isin(graves.index) & ~erros.index.isin(invalidos.index)]
print(f"total de erros: {len(erros)}")
print(f"  entre classes vizinhas (envolvem neutro): {len(vizinhos)}")
print(f"  graves (negativo <-> positivo): {len(graves)}")
print(f"  respostas inválidas: {len(invalidos)}")

# %% 6. Acurácia por tamanho do texto (4 faixas com o mesmo número de avaliações)
test_df["faixa_tamanho"] = pd.qcut(test_df.tamanho, 4, duplicates="drop")
linhas = []
for faixa, g in test_df.groupby("faixa_tamanho", observed=True):
    linhas.append({"faixa (caracteres)": str(faixa), "n": len(g), "acerto": round(g.correto.mean(), 3)})
print(pd.DataFrame(linhas).to_string(index=False))

# %% 7. Acurácia por categoria de produto (só categorias com 20 ou mais avaliações no teste)
contagem = test_df.site_category_lv1.value_counts()
linhas = []
for cat in contagem[contagem >= 20].index:
    g = test_df[test_df.site_category_lv1 == cat]
    linhas.append({"categoria": cat, "n": len(g), "acerto": round(g.correto.mean(), 3)})
print(pd.DataFrame(linhas).sort_values("acerto").to_string(index=False))

# %% 8. Exemplos de erro para LER (a parte mais importante: dá para ver se o rótulo é que está duvidoso)
pares = [("neutro", "positivo"), ("positivo", "neutro"), ("neutro", "negativo"), ("negativo", "neutro")]
for real, prev in pares:
    g = erros[(erros.label == real) & (erros.previsao == prev)]
    print(f"\n=== verdadeira: {real} | modelo disse: {prev} | {len(g)} casos (mostrando até 8) ===")
    for _, r in g.sample(min(8, len(g)), random_state=SEED).iterrows():
        print(f"  [{int(r.overall_rating)} estrelas] {r.review_text[:220]!r}")
