% Artigo técnico: fine-tuning de LLM com LoRA para polaridade de avaliações em PT-BR
% Template: IEEE Conference Template v1.1 (ieeeconf.cls)
% Versão em preto e branco, baseada no README do projeto.
% Sem arquivos de imagem externos e sem .bib: o diagrama é desenhado em TikZ
% aqui mesmo. Compile com pdflatex (duas vezes).
\documentclass[9pt]{ieeeconf}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[brazilian]{babel}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\renewcommand{\arraystretch}{1.12}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\usepackage[capitalize,nameinlink]{cleveref}
\crefname{section}{Seção}{Seções}
\Crefname{section}{Seção}{Seções}
\crefname{figure}{Figura}{Figuras}
\Crefname{figure}{Figura}{Figuras}
\crefname{table}{Tabela}{Tabelas}
\Crefname{table}{Tabela}{Tabelas}

% Links sem cor nem caixas coloridas (artigo todo em preto e branco)
\ifdefined\hypersetup\hypersetup{hidelinks}\fi

\title{Ajuste Fino com LoRA de um LLM para polaridade de avaliações de e-commerce em PT-BR}

\addauthor{Nicolle Santana}{1}
\addaffiliation{1}{Estudante de graduação -- Universidade Federal de Sergipe, São Cristóvão, Brasil}

\begin{document}

\maketitle

\begin{abstract}
Um modelo de linguagem pequeno pode ser adaptado a uma tarefa específica com poucos recursos? Ajustamos o Qwen2.5-0.5B-Instruct com LoRA para classificar a polaridade (negativa, neutra ou positiva) de avaliações de clientes em português, treinando 8,8 milhões de parâmetros (1,75\% do modelo) por uma época, com 6.000 exemplos, em cerca de 17 minutos em GPUs gratuitas. Em 600 avaliações de teste, separadas antes do treino, a acurácia passou de 61,5\% para 76,5\% e o F1 macro de 0,49 para 0,76. A classe neutra, que o modelo base nunca previa, alcançou F1 de 0,66. Dos 141 erros restantes, 96,5\% ocorrem entre classes vizinhas, e o rótulo derivado das estrelas parece ser uma das causas. O adapter (cerca de 35~MB), o código e uma demonstração online estão publicados.

\smallskip
\noindent\textbf{Palavras-chave:} ajuste fino; LoRA; análise de sentimento; português; avaliações de clientes.
\end{abstract}

\section{Introdução}
\label{sec:intro}

Classificar a polaridade de avaliações de clientes é uma tarefa comum em atendimento e comércio eletrônico. Um modelo de linguagem de uso geral pode tentar a tarefa sem treino, mas nem sempre separa bem as três classes, em especial a do meio. Técnicas de ajuste fino eficiente em parâmetros, como o LoRA \cite{hu2022lora}, permitem adaptar um modelo pequeno treinando apenas uma fração dos seus pesos, o que cabe em uma GPU gratuita.

Este trabalho percorre o ciclo completo de ajustar um modelo a uma tarefa específica: \textbf{medir o modelo base, treinar, medir de novo no mesmo teste, comparar e publicar}. A pergunta é simples: um modelo pequeno, treinado por poucos minutos com poucos exemplos, consegue aprender a distinguir avaliações negativas, neutras e positivas, inclusive a classe do meio, que o modelo base nunca previa? As contribuições são:
\begin{itemize}
  \item um procedimento de comparação com teste separado antes do treino, balanceado e sem avaliações em comum com o treino, e com a mesma instrução para os dois modelos;
  \item o ajuste fino com LoRA de um modelo de 0,5 bilhão de parâmetros, treinando 1,75\% deles;
  \item uma análise dos erros, reproduzida a partir do adapter publicado;
  \item os artefatos publicados: adapter, código, notebook com saídas e demonstração online.
\end{itemize}
Para evitar ambiguidade, usamos ``avaliação'' para a opinião escrita pelo cliente e ``teste'' para a medição do modelo.

\section{Metodologia}
\label{sec:como}

A \cref{fig:pipeline} resume as etapas, e as subseções seguintes explicam cada uma.

\begin{figure}[!t]
\centering
\begin{tikzpicture}[
  node distance=1.6mm,
  box/.style={draw, rounded corners=2pt, align=center, font=\scriptsize,
              text width=6.2cm, inner sep=2pt, minimum height=6mm},
  arr/.style={-{Stealth[length=1.4mm]}, thick}]
\node[box] (a) {\textbf{1. B2W-Reviews01:} 132.373 avaliações};
\node[box, below=of a] (b) {\textbf{2. Filtro e rótulo:} 20 a 600 caracteres; estrelas em 3 classes};
\node[box, below=of b] (c) {\textbf{3. Divisão balanceada:} 600 para teste, 6.000 para treino};
\node[box, below=of c] (d) {\textbf{4. Baseline:} modelo base, sem treino, no teste};
\node[box, below=of d] (e) {\textbf{5. Treino com LoRA:} 1 época, cerca de 17 minutos};
\node[box, below=of e] (f) {\textbf{6. Teste, comparação e publicação:} mesmo teste; adapter no Hugging Face};
\draw[arr] (a) -- (b);
\draw[arr] (b) -- (c);
\draw[arr] (c) -- (d);
\draw[arr] (d) -- (e);
\draw[arr] (e) -- (f);
\end{tikzpicture}
\caption{Etapas do projeto. O teste é separado antes do treino e permanece igual para os dois modelos.}
\label{fig:pipeline}
\end{figure}

\subsection{Dados}

Usamos o corpus B2W-Reviews01 \cite{real2019b2w}, com 132.373 avaliações de produtos da Americanas.com, coletadas entre janeiro e maio de 2018. Ficam as avaliações de 20 a 600 caracteres, e a nota em estrelas vira o rótulo: notas 1 e 2 são \emph{negativo}, nota 3 é \emph{neutro} e notas 4 e 5 são \emph{positivo}. Os dados foram baixados de uma cópia em CSV publicada no Hugging Face (\texttt{vladjr/B2W-Reviews01}), porque o conjunto original usa um \emph{script} de carregamento que a biblioteca \texttt{datasets} atual não executa.

\subsection{Divisão em treino e teste}

Separamos 600 avaliações para teste (200 por classe) e 6.000 para treino (2.000 por classe), sem avaliações em comum e com semente fixa (42). A separação é feita antes de qualquer treino.

\subsection{Modelo base e tarefa}

O modelo base é o Qwen2.5-0.5B-Instruct \cite{qwen2024}. Pedimos que ele responda com uma única palavra, usando sempre a mesma instrução: ``Você classifica a polaridade de avaliações de clientes em português. Responda apenas com uma palavra: positivo, neutro ou negativo.'' Essa instrução é usada no modelo base, no treino e no modelo ajustado. A resposta é gerada de forma determinística (decodificação gulosa), e se não começar com uma das três classes conta como inválida.

\subsection{Ajuste fino com LoRA}

O LoRA \cite{hu2022lora} mantém os pesos originais congelados e treina apenas pequenas matrizes adicionais nas camadas de atenção e do MLP. Aqui, isso significa treinar 8,8 milhões de parâmetros, 1,75\% do modelo. O resultado é um arquivo pequeno (o \emph{adapter}, cerca de 35~MB), carregado em cima do modelo base. Durante o treino, o erro é calculado só sobre a resposta (a palavra com o rótulo), e não sobre a instrução nem sobre o texto da avaliação. A \cref{tab:hiper} traz a configuração, e o treino usou as bibliotecas Transformers \cite{wolf2020} e PEFT \cite{peft}.

\begin{table}[!t]
\centering
\caption{Configuração do treino.}
\label{tab:hiper}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|>{\raggedright\arraybackslash}p{2.5cm}|>{\raggedright\arraybackslash}p{4.9cm}|}
\hline
\textbf{Item} & \textbf{Valor} \\
\hline
Modelo base & Qwen2.5-0.5B-Instruct \\ \hline
LoRA & r = 16, alpha = 32, dropout 0,05 \\ \hline
Camadas adaptadas & atenção e MLP \\ \hline
Taxa de aprendizado & 0,0002 (10 passos de aquecimento) \\ \hline
Lote & 2 por GPU, acúmulo de 8; efetivo de 32 em 2 GPUs \\ \hline
Épocas e passos & 1 época, 188 passos \\ \hline
Precisão & mista (fp16) \\ \hline
Treino e teste & 6.000 e 600 avaliações \\ \hline
Semente & 42 \\ \hline
Bibliotecas & Transformers 5.23.0, PEFT 0.21.3, Datasets 5.2.0 \\ \hline
Ambiente & GPUs gratuitas do Kaggle \\ \hline
\end{tabular}
\end{table}

O treino levou 1.014 segundos (cerca de 17 minutos), com perda média de 0,176. Com duas GPUs, o lote efetivo foi de 32 exemplos (2 por GPU, 2 GPUs e acúmulo de 8 passos), o que corresponde aos 188 passos para 6.000 exemplos.

\section{Resultados}
\label{sec:resultados}

O teste usa as 600 avaliações separadas (200 por classe). A \cref{tab:geral} mostra o resultado geral, e a \cref{tab:classe}, o resultado por classe. O F1 macro é a média do F1 das três classes.

\begin{table}[!t]
\centering
\caption{Resultado geral no teste.}
\label{tab:geral}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Modelo} & \textbf{Acurácia} & \textbf{F1 macro} & \textbf{Inválidas} \\
\hline
Base (sem treino) & 0,615 & 0,492 & 1 \\ \hline
Ajustado (LoRA) & \textbf{0,765} & \textbf{0,764} & \textbf{0} \\ \hline
\end{tabular}
\end{table}

A acurácia subiu 15,0 pontos percentuais (de 61,5\% para 76,5\%) e o F1 macro subiu 0,27 (de 0,49 para 0,76). O maior ganho foi na classe neutra: o modelo base não previu ``neutro'' em nenhuma das 600 avaliações (F1 igual a 0), e o modelo ajustado chegou a F1 de 0,66. Nas classes extremas, a precisão subiu, mas o \emph{recall} caiu (negativo de 0,96 para 0,90, positivo de 0,88 para 0,74), porque parte dos casos que o modelo base atribuía a elas passou a ser corretamente reconhecida como neutra.

\begin{table}[!t]
\centering
\caption{Precisão, \emph{recall} e F1 por classe.}
\label{tab:classe}
\begin{tabular}{|l|l|c|c|c|}
\hline
\textbf{Classe} & \textbf{Modelo} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1} \\
\hline
\multirow{2}{*}{negativo} & Base & 0,64 & 0,96 & 0,77 \\ \cline{2-5}
 & Ajustado & 0,84 & 0,90 & \textbf{0,86} \\
\hline
\multirow{2}{*}{neutro} & Base & 0,00 & 0,00 & 0,00 \\ \cline{2-5}
 & Ajustado & 0,66 & 0,66 & \textbf{0,66} \\
\hline
\multirow{2}{*}{positivo} & Base & 0,59 & 0,88 & 0,71 \\ \cline{2-5}
 & Ajustado & 0,79 & 0,74 & \textbf{0,77} \\
\hline
\end{tabular}
\end{table}

A \cref{tab:conf} mostra a matriz de confusão do modelo ajustado: cada linha é a classe verdadeira, e cada coluna, a classe prevista.

\begin{table}[!t]
\centering
\caption{Matriz de confusão do modelo ajustado.}
\label{tab:conf}
\begin{tabular}{|l|c|c|c|}
\hline
\multirow{2}{*}{\textbf{Verdadeiro}} & \multicolumn{3}{c|}{\textbf{Previsto}} \\ \cline{2-4}
 & negativo & neutro & positivo \\
\hline
negativo & \textbf{179} & 20 & 1 \\ \hline
neutro & 31 & \textbf{131} & 38 \\ \hline
positivo & 4 & 47 & \textbf{149} \\ \hline
\end{tabular}
\end{table}

\section{Análise de erros}
\label{sec:erros}

Para entender de onde vem a acurácia de 76,5\%, reconstruímos o mesmo teste, carregamos o adapter publicado no Hugging Face e analisamos os 141 erros. O resultado reproduziu a acurácia de 0,765 e a mesma matriz de confusão.

\subsection{Onde o modelo erra}

Quase todo erro é entre classes vizinhas: 136 dos 141 (96,5\%) envolvem o neutro, e só 5 são graves (4 positivas lidas como negativas e 1 negativa lida como positiva). Os erros se concentram no meio da escala: as notas 3 e 4 respondem por 101 dos 141 erros (71,6\%). A nota 4 é a mais difícil (54,3\% de acerto) e a nota 1, a mais fácil (94,2\%), como mostra a \cref{tab:notas}. Os grupos por nota são pequenos, porque o teste é balanceado por classe e não por nota.

\begin{table}[!t]
\centering
\caption{Acerto do modelo ajustado por nota em estrelas.}
\label{tab:notas}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\multirow{2}{*}{\textbf{Nota}} & \multirow{2}{*}{\textbf{Aval.}} & \multirow{2}{*}{\textbf{Acerto}} & \multicolumn{3}{c|}{\textbf{Previsto}} \\ \cline{4-6}
 & & & neg. & neu. & pos. \\
\hline
1 & 156 & 94,2\% & 147 & 8 & 1 \\ \hline
2 & 44 & 72,7\% & 32 & 12 & 0 \\ \hline
3 & 200 & 65,5\% & 31 & 131 & 38 \\ \hline
4 & 70 & 54,3\% & 2 & 30 & 38 \\ \hline
5 & 130 & 85,4\% & 2 & 17 & 111 \\ \hline
\end{tabular}
\end{table}

\subsection{O que os erros parecem ser}

Lemos 28 erros (7 de cada tipo de confusão). É uma leitura qualitativa, e não uma contagem, mas o padrão é claro:
\begin{itemize}
  \item \textbf{Rótulo discutível.} Várias avaliações de 3 estrelas têm texto claramente elogioso, como ``Muito bonito, entrega antes do prazo, chegou perfeito''. A resposta do modelo (positivo) é defensável, e quem destoa é a nota.
  \item \textbf{Perguntas no lugar de opinião.} Há avaliações que são perguntas, como ``Qual a capacidade de conexão deste modelo?'' (2 estrelas), e nelas a estrela não mede a polaridade do texto.
  \item \textbf{Avaliações mistas.} Elogio e queixa na mesma frase, como ``O produto é bonito porém de baixa qualidade'' (2 estrelas). O modelo tende ao neutro.
  \item \textbf{Reclamações com 3 estrelas.} Algumas avaliações neutras pela nota são claramente negativas no texto, e o modelo as lê como negativas.
\end{itemize}

\subsection{Um indício sobre como o modelo decide}

Contamos as avaliações com palavras de contraste (``mas'', ``porém'', ``apesar de'' e parecidas). Entre as positivas, 13 de 23 com contraste (57\%) foram lidas como neutras, contra 34 de 177 (19\%) sem contraste. Entre as neutras, o modelo errou 10 de 61 com contraste (16\%) e 59 de 139 sem contraste (42\%). O modelo parece usar essas palavras como sinal de neutralidade. É só um indício: a busca por palavras é grosseira, e apenas 23 avaliações positivas têm contraste.

\subsection{Hipóteses}

A \cref{tab:hip} resume as hipóteses examinadas e o que os dados mostram. O teto de acurácia neste conjunto provavelmente fica abaixo de 100\%, porque parte dos ``erros'' vem de rótulos discutíveis, mas esse teto não foi medido.

\begin{table}[!t]
\centering
\caption{Hipóteses sobre a origem dos erros.}
\label{tab:hip}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|>{\raggedright\arraybackslash}p{2.7cm}|>{\raggedright\arraybackslash}p{5.2cm}|}
\hline
\textbf{Hipótese} & \textbf{Evidência observada} \\
\hline
Preferência pelo neutro, por causa do treino balanceado
& Não sustentada. O modelo previu ``neutro'' 198 vezes, para 200 casos reais, e as trocas se equilibram. \\ \hline
Nota 3 ambígua e ruído nos rótulos
& Sustentada pelo padrão dos erros e pelos exemplos. A magnitude não foi medida, pois exigiria anotar uma amostra à mão. \\ \hline
Modelo pequeno e uma única época
& Não testada. \\ \hline
Tamanho do texto e categoria do produto
& Sem efeito claro. O acerto ficou entre 74,7\% e 78,2\% nas quatro faixas de tamanho, e entre 68\% e 80\% nas categorias com 20 a 102 avaliações. \\ \hline
\end{tabular}
\end{table}

\section{Exemplos de uso e demonstração}

Quatro frases testadas no modelo publicado, com a classe prevista:
\begin{itemize}
  \item ``Chegou antes do prazo, adorei! Produto excelente, recomendo muito.'' $\rightarrow$ \textbf{positivo};
  \item ``Produto chegou quebrado e o vendedor não respondeu. Péssimo.'' $\rightarrow$ \textbf{negativo};
  \item ``Produto ok, nada demais.'' $\rightarrow$ \textbf{neutro};
  \item ``Chegou antes do prazo, produto igual ao anunciado.'' $\rightarrow$ \textbf{neutro}.
\end{itemize}
São exemplos escolhidos à mão, que mostram o uso e não medem desempenho: a medida é a do teste de 600 avaliações. Uma demonstração online (Streamlit) permite colar uma avaliação e ver a classe prevista com a probabilidade de cada classe. Para rodar em CPU, ela usa precisão bfloat16 e incorpora o adapter ao modelo base, uma configuração diferente da usada nas métricas deste artigo (float32, adapter separado).

\section{Limitações}
\label{sec:limites}

\begin{itemize}
  \item Execução única, com 1 época e sem busca de hiperparâmetros.
  \item O teste tem 600 avaliações, então a margem de erro da acurácia é de cerca de 3 a 4 pontos percentuais (95\% de confiança). A melhora de 15 pontos é maior que isso, mas os números não devem ser lidos como exatos.
  \item Treino e teste são balanceados por classe, o que não reflete a proporção real de avaliações positivas, neutras e negativas.
  \item Os dados são avaliações de produtos de e-commerce de 2018. O modelo não foi testado em outros domínios, como restaurantes ou entregas, e foi treinado com textos de 20 a 600 caracteres.
  \item Os rótulos vêm da nota em estrelas, e a diferença entre notas vizinhas é subjetiva, principalmente no neutro.
  \item Usamos uma única instrução; a sensibilidade a outras formulações não foi testada.
  \item A leitura dos erros foi qualitativa e sem segundo leitor.
  \item O modelo é pequeno, e erros em ironia, gírias e textos ambíguos são esperados. Não foi feita análise por grupos ou regiões.
  \item Não houve comparação com modelos mais simples (um modelo clássico e um BERT) nas mesmas 600 avaliações, que seria o próximo passo natural, junto com a anotação à mão de uma amostra de erros.
\end{itemize}

\section{Conclusão}
\label{sec:conclusao}

Um modelo de 0,5 bilhão de parâmetros, ajustado com LoRA em cerca de 17 minutos e com 6.000 exemplos, passou de 61,5\% para 76,5\% de acurácia e de 0,49 para 0,76 de F1 macro na classificação de polaridade em três classes, e passou a reconhecer a classe neutra. Os erros restantes ocorrem principalmente entre classes vizinhas e nas notas intermediárias, onde o rótulo por estrelas é mais discutível. Como próximos passos, propomos anotar à mão uma amostra de erros para estimar o ruído dos rótulos e comparar com um modelo clássico e com um BERT nas mesmas 600 avaliações.

\section*{Disponibilidade e licenças}

O código, o \emph{notebook} com todas as saídas, o \texttt{resultados.json}, o \emph{script} de análise de erros e o \texttt{requirements.txt} estão em \url{https://github.com/NS-0111/finetune-llm-polaridade-ptbr}. O adapter está em \url{https://huggingface.co/nicollesantana/qwen2.5-0.5b-polaridade-ptbr-lora}, e a demonstração, em \url{https://polaridade-ptbr.streamlit.app/}. O código tem licença MIT. O adapter é CC BY-NC-SA 4.0, por precaução com a licença do conjunto de dados, cuja página no Hugging Face indica CC BY 4.0, mas cujo registro indica CC BY-NC-SA 4.0. O modelo base tem licença Apache 2.0.

\section*{Uso de ferramentas de IA}

Um assistente de IA (Claude) foi usado como apoio para tirar dúvidas durante o desenvolvimento do projeto.

\begin{thebibliography}{9}
\bibitem{real2019b2w} L.~Real, M.~Oshiro, and A.~Mafra, ``B2W-reviews01: an open product reviews corpus,'' in \textit{Proc. XII Symp. Information and Human Language Technology (STIL)}, 2019.
\bibitem{hu2022lora} E.~J. Hu \textit{et al.}, ``LoRA: Low-rank adaptation of large language models,'' in \textit{Proc. Int. Conf. Learning Representations (ICLR)}, 2022.
\bibitem{qwen2024} Qwen Team, ``Qwen2.5 technical report,'' arXiv:2412.15115, 2024.
\bibitem{wolf2020} T.~Wolf \textit{et al.}, ``Transformers: State-of-the-art natural language processing,'' in \textit{Proc. EMNLP: System Demonstrations}, 2020, pp.~38--45.
\bibitem{peft} S.~Mangrulkar \textit{et al.}, ``PEFT: State-of-the-art parameter-efficient fine-tuning methods,'' 2022. [Online]. Available: \url{https://github.com/huggingface/peft}
\end{thebibliography}

\end{document}
