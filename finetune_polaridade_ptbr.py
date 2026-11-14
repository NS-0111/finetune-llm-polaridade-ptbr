# %% [markdown]
# # Fine-tuning de LLM (LoRA) para polaridade de avaliações de clientes em PT-BR
# Rode célula por célula no Kaggle (GPU T4) ou no Colab. Cada "# %%" é uma célula.

# %% 0. Instalação (descomente no Kaggle/Colab)
# !pip install -q -U transformers peft datasets accelerate scikit-learn
# No Kaggle, se o peft reclamar de versão do torchao, rode antes de treinar:
# !pip uninstall -y torchao

# %% 1. Configuração
import json, random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # pode trocar por outro modelo pequeno com PT-BR
N_TRAIN, N_TEST = 6000, 600                 # amostras balanceadas por classe
LABELS = ["negativo", "neutro", "positivo"]

# %% 2. Dados: avaliações B2W (e-commerce, PT-BR) -> 3 classes
# O dataset oficial no HF usa script de carregamento (não suportado mais), então baixamos o CSV de uma cópia.
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="vladjr/B2W-Reviews01", filename="B2W-Reviews01.csv", repo_type="dataset")
raw = pd.read_csv(path)
print(raw.shape)
print(raw.columns.tolist())  # confira: esperamos review_text e overall_rating

df = raw[["review_text", "overall_rating"]].dropna()
df = df[df.review_text.str.len().between(20, 600)]
df["label"] = df.overall_rating.astype(int).map(
    lambda r: "negativo" if r <= 2 else ("neutro" if r == 3 else "positivo"))

def sample_balanced(frame, n_total):
    per_class = n_total // len(LABELS)
    parts = [frame[frame.label == lab].sample(per_class, random_state=SEED) for lab in LABELS]
    return pd.concat(parts).sample(frac=1, random_state=SEED)

test_df = sample_balanced(df, N_TEST)                  # teste fixo, nunca usado no treino
train_df = sample_balanced(df.drop(test_df.index), N_TRAIN)
print(train_df.label.value_counts(), test_df.label.value_counts())

# %% 3. Tokenizer e prompt
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

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
def predict(model, texts, bs=32):
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(texts), bs):
        batch = [build_prompt(t) for t in texts[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += [parse(x) for x in tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    tok.padding_side = "right"
    return preds

def evaluate(model, name):
    preds = predict(model, test_df.review_text.tolist())
    y = test_df.label.tolist()
    res = {"modelo": name,
           "accuracy": accuracy_score(y, preds),
           "macro_f1": f1_score(y, preds, average="macro", labels=LABELS),
           "invalidos": sum(p == "invalido" for p in preds)}
    print(res)
    print(classification_report(y, preds, labels=LABELS, zero_division=0))
    print(confusion_matrix(y, preds, labels=LABELS))
    return res

# %% 4. Baseline: modelo base, sem fine-tuning (zero-shot)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to("cuda")
res_base = evaluate(model, "base (zero-shot)")

# %% 5. Dataset de treino (loss só na resposta)
def encode(row):
    prompt_ids = tok(build_prompt(row["review_text"]), add_special_tokens=False).input_ids
    answer_ids = tok(row["label"] + tok.eos_token, add_special_tokens=False).input_ids
    return {"input_ids": prompt_ids + answer_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
            "labels": [-100] * len(prompt_ids) + answer_ids}

train_ds = Dataset.from_pandas(train_df[["review_text", "label"]].reset_index(drop=True))
train_ds = train_ds.map(encode, remove_columns=train_ds.column_names)

# %% 6. LoRA + treino
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(output_dir="out", per_device_train_batch_size=2,
                         gradient_accumulation_steps=8, learning_rate=2e-4,
                         num_train_epochs=1, lr_scheduler_type="cosine", warmup_steps=10,
                         logging_steps=25, save_strategy="no", fp16=True,
                         report_to="none", seed=SEED)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
trainer.train()

# %% 7. Avaliação do modelo ajustado (mesmo teste fixo)
res_ft = evaluate(model, "fine-tuned (LoRA)")
json.dump([res_base, res_ft], open("resultados.json", "w"), indent=2, ensure_ascii=False)

# %% 8. Publicar no Hugging Face (usuário HF: nicollesantana)
# from huggingface_hub import notebook_login; notebook_login()
# model.push_to_hub("nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora")
# tok.push_to_hub("nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora")
