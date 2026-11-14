# Fine-tuning de LLM para polaridade de avaliações em português

Ajuste fino com LoRA do [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) para classificar a **polaridade** (negativo, neutro ou positivo) de avaliações de clientes em português do Brasil. O modelo recebe o texto da avaliação e responde com uma única palavra.

**Modelo publicado:** [nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora](https://huggingface.co/nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora)

## Resultado

Avaliação em 600 avaliações de teste (200 por classe), separadas antes do treino:

| Modelo | Acurácia | F1 macro | Respostas inválidas |
|---|---|---|---|
| Base (zero-shot) | 0,615 | 0,492 | 1 |
| Ajustado (LoRA) | 0,765 | 0,764 | 0 |

O modelo base não previu a classe "neutro" em nenhuma das 600 avaliações (F1 = 0). Depois do ajuste, o F1 do neutro foi para 0,66. O relatório completo por classe e a matriz de confusão estão no [model card](https://huggingface.co/nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora) e no `resultados.json`.

## Como funciona

1. **Dados:** o corpus [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), com 132.373 avaliações de produtos da Americanas.com. Foram mantidas avaliações de 20 a 600 caracteres, e a nota em estrelas virou o rótulo (1 e 2 = negativo, 3 = neutro, 4 e 5 = positivo).
2. **Divisão:** 600 avaliações para teste e 6.000 para treino, balanceadas por classe e sem avaliações em comum (semente 42).
3. **Baseline:** o modelo base é avaliado no teste, sem treino, com o mesmo prompt.
4. **Treino:** LoRA (r = 16, alpha = 32) em todas as camadas lineares da atenção e do MLP, 1 época, perda calculada só na resposta. Foram treinados 8,8 milhões de parâmetros, 1,75% do modelo.
5. **Avaliação:** o modelo ajustado é avaliado no mesmo teste, e as métricas são salvas em `resultados.json`.

## Tecnologias utilizadas

- **Python**: linguagem de todo o projeto
- **Qwen2.5-0.5B-Instruct**: modelo de linguagem base que foi ajustado
- **PyTorch**: execução do modelo e do treino na GPU
- **Hugging Face Transformers** (5.23.0): carregamento do modelo e do tokenizer, prompt de chat e `Trainer` para o treino
- **PEFT** (0.21.3): LoRA, que treina só 1,75% dos parâmetros do modelo
- **Hugging Face Datasets** (5.2.0): construção do conjunto de treino
- **Accelerate**: usado pelo `Trainer` para gerenciar o treino na GPU
- **Hugging Face Hub**: download do CSV dos dados e publicação do adapter
- **scikit-learn**: métricas de avaliação (acurácia, F1, relatório por classe e matriz de confusão)
- **pandas** e **NumPy**: filtragem, rotulagem e sorteio das amostras
- **Kaggle Notebooks**: ambiente de execução, com GPU gratuita
- **Git e GitHub**: versionamento e publicação do código

## Arquivos

| Arquivo | O que é |
|---|---|
| `finetune_polaridade_ptbr.py` | Script limpo, dividido em células (`# %%`) |
| `finetune_llm_polaridade_ptbr.ipynb` | Execução completa no Kaggle, com as saídas |
| `resultados.json` | Métricas do modelo base e do ajustado |
| `requirements.txt` | Bibliotecas usadas |

## Como reproduzir

Precisa de uma GPU. O treino levou cerca de 17 minutos em GPU gratuita do Kaggle.

1. Abra um notebook no Kaggle (ou no Colab) com GPU e internet ligadas.
2. Instale as dependências: `pip install -r requirements.txt`.
3. Copie as células do `finetune_polaridade_ptbr.py`, uma por vez, e rode em ordem.
4. No Kaggle, se o `peft` reclamar da versão do `torchao`, rode `pip uninstall -y torchao` antes do treino.

Os dados são baixados de uma cópia em CSV do B2W-Reviews01 publicada no Hugging Face, porque o dataset original usa um script de carregamento que a biblioteca `datasets` atual não executa.

## Limitações

- Execução única, com 1 época e sem busca de hiperparâmetros.
- O teste tem 600 avaliações, então a margem de erro da acurácia é de cerca de 3 a 4 pontos percentuais.
- O rótulo vem das estrelas, e a diferença entre notas vizinhas é subjetiva, principalmente no neutro.
- Os dados são avaliações de produtos de e-commerce de 2018. O modelo não foi avaliado em outros domínios, como avaliações de restaurantes.

## Licenças e créditos

- O adapter publicado segue **CC BY-NC-SA 4.0**, por precaução com a licença do dataset original. A página do dataset no Hugging Face indica CC BY 4.0, mas o registro do corpus indica CC BY-NC-SA 4.0.
- Modelo base: Qwen2.5-0.5B-Instruct, licença Apache 2.0.
- Dados: B2W-Reviews01, de Real, Oshiro e Mafra (2019), *B2W-reviews01: an open product reviews corpus*, STIL.
