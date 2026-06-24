# IEEE Seminar Paper (LaTeX, Croatian) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a Croatian IEEE-conference seminar paper (`seminar.tex`) that strictly follows the shipped `ieeeconf` template and reports the project's actual implementation and `results/`.

**Architecture:** One self-contained `seminar.tex` placed next to `ieeeconf.cls`. Built section-by-section in the fixed IEEE order. Croatian enabled via additive preamble packages (no margin/format changes). The single authored figure is a TikZ data-flow diagram; the other three figures are existing PNGs from `results/`. Numbers come verbatim from `results/80k/summary.txt` and `results/comparison/summary.md`.

**Tech Stack:** LaTeX (`ieeeconf` document class), `babel` (croatian), `graphicx`, `amsmath`, `booktabs`, `tikz`. Target compiler: pdfLaTeX.

## Global Constraints

- File: `Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/seminar.tex`. **Do not modify `main.tex`** (reference template) or anything in `results/`.
- Document class verbatim: `\documentclass[letterpaper, 10pt, conference]{ieeeconf}` + `\IEEEoverridecommandlockouts` + `\overrideIEEEmargins`. **Never** change margins or class options.
- Language: **Croatian** prose with proper diacritics (š, č, ć, ž, đ); algorithm names kept in original form.
- Headline metric is **ROC-AUC** (ranking quality). The label-free MAD cutoff over-flags vs the 0.17% base rate, so thresholded F1/precision are low **by design** — frame results this way.
- `MAD_K = 3.0`, `DEFAULT_NU = 0.05`, `RANDOM_STATE = 42`.
- Author block stays a **placeholder** (user fills in).
- Every reported number must trace to `results/` — no invented figures.
- **No local LaTeX toolchain exists.** Per-task verification = structural lint (commands given below). The user compiles the final PDF (Overleaf or MiKTeX).
- Commit messages: short, concise, no `Co-Authored-By` / `Claude-Session` trailers.

## Verification helper (used by every task)

There is no `pdflatex`, so each task verifies with these three structural checks, run from the repo root via the Bash tool:

```bash
cd "Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia"
# 1) environment balance — counts must be equal
echo "begin=$(grep -c '\\begin{' seminar.tex) end=$(grep -c '\\end{' seminar.tex)"
# 2) every \cite key has a matching \bibitem
for k in $(grep -oE '\\cite\{[^}]+\}' seminar.tex | sed -E 's/\\cite\{|\}//g' | tr ',' '\n' | sort -u); do
  grep -q "\\bibitem{$k}" seminar.tex && echo "OK cite $k" || echo "MISSING bibitem $k"; done
# 3) every \includegraphics path exists (paths are relative to this folder)
for p in $(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' seminar.tex | sed -E 's/.*\{([^}]+)\}/\1/'); do
  [ -f "$p" ] && echo "OK img $p" || echo "MISSING img $p"; done
```

Expected at completion: `begin == end`, every cite `OK`, every img `OK`.

## File Structure

- **Create:** `Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/seminar.tex` — the entire paper (single file, like `main.tex`).
- **Read-only inputs:** `results/80k/summary.txt`, `results/comparison/summary.md`, `results/comparison/rocauc_vs_size.png`, `results/comparison/runtime_vs_size.png`, `results/80k/metrics_bar.png`, `src/*.py` (for cross-checking claims), `ieeeconf.cls` (must sit beside `seminar.tex`).

Figure include paths (relative to the IEEE folder): `../results/comparison/rocauc_vs_size.png`, `../results/comparison/runtime_vs_size.png`, `../results/80k/metrics_bar.png`.

---

### Task 1: Document skeleton, preamble, title/author, empty sections

**Files:**
- Create: `Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/seminar.tex`

**Interfaces:**
- Produces: a compiling skeleton with all section headers in Croatian and an empty `thebibliography`. Later tasks fill section bodies and add `\bibitem`s/`\cite`s.

- [ ] **Step 1: Create `seminar.tex` with preamble + title + placeholder author + empty sections**

```latex
\documentclass[letterpaper, 10pt, conference]{ieeeconf}
\IEEEoverridecommandlockouts
\overrideIEEEmargins

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[croatian]{babel}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}

\title{\LARGE \bf ML5 --- Strojno učenje za detekciju anomalija \\[2pt]
\large Machine Learning for Anomaly Detection}

\author{Ime Prezime$^{1}$%
\thanks{$^{1}$Autor je student na Tehničkom fakultetu, Sveučilište u Rijeci,
kolegij Strojno učenje (ML5). {\tt\small ime.prezime@email.com}}%
}

\begin{document}
\maketitle
\thispagestyle{empty}
\pagestyle{empty}

\begin{abstract}
% TASK 9
\end{abstract}

\section{\textbf{UVOD}}
% TASK 3

\section{\textbf{METODOLOGIJA}}
% TASK 4 (prose + Table I + MAD eq) and TASK 5 (Fig. 1 TikZ)

\section{\textbf{STUDIJA SLUČAJA}}
% TASK 6
\subsection{Skup podataka}
\subsection{Vrednovanje}

\section{\textbf{REZULTATI I RASPRAVA}}
% TASK 7 (tables + figures) and TASK 8 (discussion)

\section{\textbf{ZAKLJUČAK}}
% TASK 9

\begin{thebibliography}{99}
% TASK 2
\end{thebibliography}

\end{document}
```

Note: the `% TASK N` comments are scaffolding markers; they are removed as each section is filled. They are not placeholders in the deliverable sense — every task below replaces its marker with real content.

- [ ] **Step 2: Verify structure**

Run the Verification helper. Expected: `begin == end` (abstract, document, thebibliography all balanced), no cites yet, no images yet.

- [ ] **Step 3: Commit**

```bash
git add "Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/seminar.tex"
git commit -m "docs: scaffold seminar.tex (preamble, title, empty sections)"
```

---

### Task 2: Bibliography (9 references)

**Files:**
- Modify: `seminar.tex` (replace the `% TASK 2` marker inside `thebibliography`)

**Interfaces:**
- Produces cite keys used by later tasks: `agyemang`, `kaggle`, `sklearn`, `ocsvm`, `iforest`, `lof`, `mcd`, `mad`, `pytorch`.

- [ ] **Step 1: Insert the 9 `\bibitem` entries**

```latex
\bibitem{agyemang} E. F. Agyemang, ``Anomaly detection using unsupervised machine learning algorithms: A simulation study,'' \emph{Scientific African}, vol.~26, e02386, 2024.
\bibitem{kaggle} Machine Learning Group, Université Libre de Bruxelles, ``Credit Card Fraud Detection,'' Kaggle, 2018. [Online]. Available: \texttt{https://www.kaggle.com/mlg-ulb/creditcardfraud}
\bibitem{sklearn} F. Pedregosa \emph{et al.}, ``Scikit-learn: Machine learning in Python,'' \emph{J. Mach. Learn. Res.}, vol.~12, pp.~2825--2830, 2011.
\bibitem{ocsvm} B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, ``Estimating the support of a high-dimensional distribution,'' \emph{Neural Comput.}, vol.~13, no.~7, pp.~1443--1471, 2001.
\bibitem{iforest} F. T. Liu, K. M. Ting, and Z.-H. Zhou, ``Isolation forest,'' in \emph{Proc. 8th IEEE Int. Conf. Data Mining (ICDM)}, 2008, pp.~413--422.
\bibitem{lof} M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, ``LOF: Identifying density-based local outliers,'' in \emph{Proc. ACM SIGMOD Int. Conf. Management of Data}, 2000, pp.~93--104.
\bibitem{mcd} P. J. Rousseeuw and K. Van Driessen, ``A fast algorithm for the minimum covariance determinant estimator,'' \emph{Technometrics}, vol.~41, no.~3, pp.~212--223, 1999.
\bibitem{mad} B. Iglewicz and D. C. Hoaglin, \emph{How to Detect and Handle Outliers}. Milwaukee, WI: ASQC Quality Press, 1993.
\bibitem{pytorch} A. Paszke \emph{et al.}, ``PyTorch: An imperative style, high-performance deep learning library,'' in \emph{Adv. Neural Inf. Process. Syst. 32 (NeurIPS)}, 2019, pp.~8024--8035.
```

- [ ] **Step 2: Verify**

Run the Verification helper. Expected: `begin == end` unchanged; still no cites (added in later tasks); 9 `\bibitem`s present (`grep -c '\\bibitem' seminar.tex` → 9).

- [ ] **Step 3: Commit**

```bash
git add "Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/seminar.tex"
git commit -m "docs: add 9-reference bibliography to seminar"
```

---

### Task 3: Introduction (UVOD)

**Files:**
- Modify: `seminar.tex` (replace `% TASK 3`)

**Interfaces:**
- Consumes cite keys: `agyemang`, `kaggle` (and may cite `iforest`, `ocsvm`, `lof`).

**Content brief** (write as Croatian prose, ~3–4 paragraphs):
1. **Problem & motivation:** kreditne kartice i prijevare; ekstremna neravnoteža (0,17% prijevara, ~1:577), zbog čega je točnost (accuracy) zavaravajuća mjera; potreba za detekcijom anomalija.
2. **Why unsupervised:** oznake (labels) su skupe/rijetke u praksi; nenadzirani pristup uči "normalno" i označava odstupanja. Spomenuti tipove pristupa (granica, gustoća, izolacija, rekonstrukcija) uz citate `\cite{ocsvm,iforest,lof}`.
3. **Reference paper gap:** Agyemang (2024) `\cite{agyemang}` je **simulacijska studija** na 220 sintetičkih 2-D točaka; njegova ograničenja izrijekom traže validaciju na stvarnim podacima — to je motivacija ovog rada.
4. **Goal & contribution:** primijeniti identičan skup algoritama i hiperparametara na velik, neuravnotežen, visoko-dimenzijski stvarni skup `\cite{kaggle}`; uvesti **label-free MAD prag**; provesti **sweep veličine uzorka** (30k–230k); glavna mjera ROC-AUC. Najaviti glavni nalaz: jaki modeli odlično rangiraju prijevaru, ali pragovanje bez oznaka je usko grlo.

- [ ] **Step 1:** Replace `% TASK 3` with the Croatian Introduction prose following the brief, inserting `\cite{...}` at the points above.
- [ ] **Step 2: Verify** — run helper; expected `begin == end`, cites `agyemang`, `kaggle`, `ocsvm`, `iforest`, `lof` all `OK`.
- [ ] **Step 3: Commit** — `git commit -m "docs: write Introduction (UVOD)"`

---

### Task 4: Methodology prose + Table I + MAD equation

**Files:**
- Modify: `seminar.tex` (start of `% TASK 4` region under METODOLOGIJA)

**Interfaces:**
- Consumes cites: `ocsvm`, `iforest`, `lof`, `mcd`, `pytorch`, `mad`, `sklearn`.
- Produces label `tab:lineup` (Table I), equation label `eq:mad`. Task 5 adds `fig:pipeline` in the same section.

**Content brief** (Croatian prose + one table + one equation):
- Para A — **Pregled modela:** kratko opisati 6 nenadziranih detektora i nadzirani baseline, svaki 1–2 rečenice s mehanizmom: One-Class SVM (RBF granica) `\cite{ocsvm}`, OCSVM-SGD (linearna SGD aproksimacija), Isolation Forest (slučajna izolacija) `\cite{iforest}`, LOF (lokalna gustoća) `\cite{lof}`, Robust Covariance / Elliptic Envelope (Gaussova elipsa, MCD) `\cite{mcd}`, Autoencoder (rekonstrukcijska pogreška, PyTorch/GPU) `\cite{pytorch}`, Logistic Regression (nadzirani baseline) `\cite{sklearn}`. Uputiti na Tablicu~\ref{tab:lineup}.
- Para B — **Ujednačeni anomaly score:** svi detektori svedeni na "veće = anomalnije" negiranjem `decision_function` (i `negative_outlier_factor_` za LOF). Ova konvencija predznaka je nužna da ROC-AUC ne invertira.
- Para C — **Label-free MAD prag:** robusni modificirani z-score; jednadžba `eq:mad`; prag `k = MAD_K = 3.0` `\cite{mad}`; udio označenih = podacima-vođena kontaminacija; stvarna stopa prijevare ne ulazi u detekciju (samo u izvještavanje).
- Para D — **Mjere:** ROC-AUC na kontinuiranim ocjenama (neovisna o pragu, primarna) + Accuracy, Precision, Recall, F1, R² na binarnim predikcijama; "implied contamination" = udio označenih.

- [ ] **Step 1: Insert Table I (algorithm lineup + hyperparameters)**

```latex
\begin{table}[t]
\caption{Skup algoritama i ključni hiperparametri. Prvih šest su nenadzirani; Logistička regresija je nadzirani baseline.}
\label{tab:lineup}
\centering
\begin{tabular}{@{}lll@{}}
\toprule
Model & Hiperparametri & Ocjena anomalije \\
\midrule
One-Class SVM        & RBF, $\gamma{=}0.1$, $\nu{=}0.05$        & $-$\,decision\_function \\
One-Class SVM (SGD)  & $\nu{=}0.05$, max\_iter${=}1000$         & $-$\,decision\_function \\
Isolation Forest     & $n{=}100$, contamination${=}$auto       & $-$\,decision\_function \\
Local Outlier Factor & $k{=}20$, novelty${=}$False             & $-$\,neg.\ outlier factor \\
Robust Covariance    & contamination${=}0.1$                   & $-$\,decision\_function \\
Autoencoder (GPU)    & $(20,14)$, 30 ep., bs${=}2048$          & rekonstr.\ MSE \\
Logistic Regression  & class\_weight${=}$balanced              & predict\_proba \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2: Insert the MAD equation** (within Para C)

```latex
\begin{equation}
z_i = \frac{s_i - \mathrm{med}(s)}{1.4826 \cdot \mathrm{MAD}(s)}, \qquad
\text{anomalija} \iff z_i > k, \quad k = 3.0
\label{eq:mad}
\end{equation}
```

- [ ] **Step 3:** Write Paras A–D (Croatian) around the table and equation, with the cites listed above.
- [ ] **Step 4: Verify** — run helper; expected `begin == end` (table added a balanced `table`+`tabular`, equation is not a begin/end pair issue since `equation` is balanced), cites `mcd`, `pytorch`, `mad`, `sklearn` now `OK`.
- [ ] **Step 5: Commit** — `git commit -m "docs: write Methodology prose, lineup table, MAD equation"`

---

### Task 5: Fig. 1 — TikZ data-flow diagram

**Files:**
- Modify: `seminar.tex` (insert figure in METODOLOGIJA, after Para A or at section end)

**Interfaces:**
- Produces label `fig:pipeline`, referenced by Methodology prose (`\ref{fig:pipeline}`).

- [ ] **Step 1: Insert the TikZ figure**

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  font=\scriptsize, node distance=3.5mm,
  box/.style={draw, rounded corners, align=center, inner sep=2pt,
              text width=2.6cm, minimum height=6mm},
  arr/.style={-{Stealth[length=2mm]}}]
\node[box] (data) {Sirovi podaci\\ \texttt{creditcard.csv}\\ (284\,807)};
\node[box, below=of data] (prep) {Pretprocesiranje\\ izbaci \texttt{Time};\\ skaliraj \texttt{Amount}};
\node[box, below=of prep] (samp) {Slučajni poduzorak\\ 30k--230k ($\sim$0.17\%)};
\node[box, below=of samp] (model) {Nenadzirani model\\ (1 od 6)};
\node[box, below=of model] (score) {Anomaly score\\ veće $=$ anomalnije};
\node[box, below=of score] (mad) {MAD prag $k{=}3.0$\\ (bez oznaka)};
\node[box, below=of mad] (met) {Metrike: ROC-AUC,\\ F1, R², \dots};
\node[box, right=8mm of model] (split) {Stratificirana\\ podjela 70/30};
\node[box, below=of split] (lr) {Logistička\\ regresija};
\node[box, below=of lr] (met2) {Metrike na\\ 30\% skupu};
\draw[arr] (data) -- (prep);
\draw[arr] (prep) -- (samp);
\draw[arr] (samp) -- (model);
\draw[arr] (model) -- (score);
\draw[arr] (score) -- (mad);
\draw[arr] (mad) -- (met);
\draw[arr] (samp.east) -- ++(0.4,0) |- (split.west);
\draw[arr] (split) -- (lr);
\draw[arr] (lr) -- (met2);
\end{tikzpicture}
\caption{Tok podataka. Nenadzirani modeli (lijevo) ocjenjuju se na cijelom poduzorku bez podjele; nadzirani baseline (desno) uči na 70\% i vrednuje se na 30\%.}
\label{fig:pipeline}
\end{figure}
```

Note: TikZ `positioning` placement may need a small tweak after the user's first compile (e.g. node `text width` or `node distance`); the layout is intentionally simple/linear to minimize that risk.

- [ ] **Step 2: Verify** — run helper; expected `begin == end` (figure + tikzpicture balanced).
- [ ] **Step 3: Commit** — `git commit -m "docs: add TikZ data-flow diagram (Fig. 1)"`

---

### Task 6: Case Study (STUDIJA SLUČAJA) — Datasets + Evaluation + Table II

**Files:**
- Modify: `seminar.tex` (`% TASK 6` region; fill the two `\subsection`s)

**Interfaces:**
- Consumes cites: `kaggle`. Produces label `tab:dataset` (Table II).

**Content brief:**
- **A. Skup podataka** (prose + Table II): Kaggle Credit Card Fraud `\cite{kaggle}`; 284\,807 transakcija, 492 prijevare (0,17%); `V1`–`V28` su PCA-anonimizirane značajke, `Amount` (skalira se), `Time` (izbacuje se), `Class` (samo za evaluaciju). Pretprocesiranje: izbaci `Time`, `StandardScaler` na `Amount`, zadrži duplikate. Zašto **zajednički poduzorak**: One-Class SVM/LOF/Elliptic Envelope skaliraju ~O(n²); zašto **sweep** 30k/80k/150k/230k uz očuvanje prirodne neravnoteže. Dva režima: nenadzirani na cijelom poduzorku (bez podjele), Logistic Regression na 70/30 stratificiranoj podjeli. Uputiti na Tablicu~\ref{tab:dataset}.
- **B. Vrednovanje** (prose): 6 mjera; zašto je **ROC-AUC** primarna (neovisna o pragu); implied contamination kao dijagnostika preflagiranja; `RANDOM_STATE = 42` za ponovljivost.

- [ ] **Step 1: Insert Table II (dataset composition)**

```latex
\begin{table}[t]
\caption{Sastav skupa podataka (Kaggle Credit Card Fraud).}
\label{tab:dataset}
\centering
\begin{tabular}{@{}ll@{}}
\toprule
Svojstvo & Vrijednost \\
\midrule
Ukupno transakcija     & 284\,807 \\
Normalne (Class${=}0$) & 284\,315 (99,83\%) \\
Prijevare (Class${=}1$)& 492 (0,17\%) \\
Značajke za model      & 30 (V1--V28, Amount) \\
Omjer neravnoteže      & $\approx 1:577$ \\
Nedostajuće vrijednosti& 0 \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2:** Write subsections A and B (Croatian) per the brief.
- [ ] **Step 3: Verify** — run helper; expected balance OK, `kaggle` cite `OK`.
- [ ] **Step 4: Commit** — `git commit -m "docs: write Case Study (datasets, evaluation, dataset table)"`

---

### Task 7: Results — Table III + cross-size Tables IV–VI + Table VII + 3 figures

**Files:**
- Modify: `seminar.tex` (`% TASK 7` region under REZULTATI I RASPRAVA)

**Interfaces:**
- Produces labels `tab:main80k`, `tab:auc`, `tab:f1`, `tab:runtime`, `tab:agyemang`, `fig:auc`, `fig:runtime`, `fig:bars`. Task 8 references these in prose.

Numbers below are copied from `results/80k/summary.txt` and `results/comparison/summary.md`. **Cross-check before committing** (open both files).

- [ ] **Step 1: Insert Table III (main 80k metrics)** — use `table*` (spans both columns; 8 columns wide).

```latex
\begin{table*}[t]
\caption{Mjere na zajedničkom poduzorku od 80\,000 redaka (127 prijevara). Prvih šest su nenadzirani; Logistička regresija je nadzirani baseline na 30\% skupu.}
\label{tab:main80k}
\centering
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Model & Točnost & Preciznost & Odziv & F1 & ROC-AUC & R$^2$ & Impl.\ kontam. \\
\midrule
One-Class SVM         & 99,29\% & 10,14\% & 44,09\% & \textbf{16,49\%} & \textbf{0,9635} & $-3{,}47$  & 0,69\% \\
One-Class SVM (SGD)   & 99,35\% & 0,00\%  & 0,00\%  & 0,00\%  & 0,0504 & $-3{,}11$   & 0,49\% \\
Isolation Forest      & 95,60\% & 2,97\%  & 84,25\% & 5,73\%  & 0,9612 & $-26{,}75$  & 4,51\% \\
Local Outlier Factor  & 92,77\% & 0,44\%  & 19,69\% & 0,86\%  & 0,5481 & $-44{,}63$  & 7,14\% \\
Robust Covariance     & 61,08\% & 0,38\%  & 92,91\% & 0,75\%  & 0,8882 & $-244{,}53$ & 39,05\% \\
Autoencoder (GPU)     & 92,55\% & 1,82\%  & 86,61\% & 3,56\%  & 0,9523 & $-46{,}01$  & 7,57\% \\
Logistic Regression   & 97,30\% & 5,15\%  & 92,11\% & 9,76\%  & 0,9606 & $-16{,}05$  & 2,83\% \\
\bottomrule
\end{tabular}
\end{table*}
```

- [ ] **Step 2: Insert Table IV (ROC-AUC across sizes)** — single column.

```latex
\begin{table}[t]
\caption{ROC-AUC po veličini uzorka (neovisno o pragu).}
\label{tab:auc}
\centering
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & 30k & 80k & 150k & 230k \\
\midrule
Isolation Forest      & \textbf{0,9809} & 0,9612 & 0,9489 & 0,9467 \\
One-Class SVM         & 0,9796 & 0,9635 & 0,9470 & 0,9430 \\
Autoencoder (GPU)     & 0,9718 & 0,9523 & 0,9325 & 0,9432 \\
Logistic Regression   & 0,9208 & 0,9606 & 0,9803 & 0,9770 \\
Robust Covariance     & 0,9389 & 0,8882 & 0,9250 & 0,9191 \\
Local Outlier Factor  & 0,7696 & 0,5481 & 0,5052 & 0,5099 \\
One-Class SVM (SGD)   & 0,0185 & 0,0504 & 0,1389 & 0,3668 \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 3: Insert Table V (F1 across sizes)** — single column.

```latex
\begin{table}[t]
\caption{F1 po veličini uzorka (MAD prag preflagira pa su vrijednosti niske).}
\label{tab:f1}
\centering
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & 30k & 80k & 150k & 230k \\
\midrule
One-Class SVM         & 0,00\%  & 16,49\% & 15,94\% & 15,26\% \\
Logistic Regression   & 15,49\% & 9,76\%  & 8,60\%  & 9,80\% \\
Isolation Forest      & 5,10\%  & 5,73\%  & 5,89\%  & 5,47\% \\
Autoencoder (GPU)     & 5,20\%  & 3,56\%  & 2,57\%  & 3,23\% \\
One-Class SVM (SGD)   & 0,00\%  & 0,00\%  & 1,22\%  & 7,28\% \\
Local Outlier Factor  & 1,28\%  & 0,86\%  & 0,68\%  & 0,64\% \\
Robust Covariance     & 0,71\%  & 0,75\%  & 0,75\%  & 0,72\% \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 4: Insert Table VI (total runtime across sizes)** — single column. (Per spec, this may be dropped if space-constrained; keep for now.)

```latex
\begin{table}[t]
\caption{Ukupno vrijeme izvođenja (s) po veličini uzorka.}
\label{tab:runtime}
\centering
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & 30k & 80k & 150k & 230k \\
\midrule
One-Class SVM (SGD)   & 0,03  & 0,07   & 0,17   & 0,28 \\
Logistic Regression   & 0,09  & 0,17   & 0,35   & 0,57 \\
Isolation Forest      & 0,30  & 0,47   & 1,07   & 1,67 \\
Autoencoder (GPU)     & 2,67  & 2,63   & 5,25   & 6,82 \\
Robust Covariance     & 2,76  & 6,62   & 16,11  & 24,31 \\
Local Outlier Factor  & 3,55  & 5,92   & 25,39  & 57,15 \\
One-Class SVM         & 16,52 & 107,13 & 370,36 & \textbf{893,62} \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 5: Insert Table VII (comparison to Agyemang, F1 for outliers)** — single column.

```latex
\begin{table}[t]
\caption{Usporedba F1 za izdvojene točke: referentni rad (220 točaka, 2-D) naspram ovog rada (80k poduzorak, 30-D).}
\label{tab:agyemang}
\centering
\begin{tabular}{@{}lrr@{}}
\toprule
Model & Agyemang \cite{agyemang} & Ovaj rad \\
\midrule
One-Class SVM        & 66,67\% & 16,49\% \\
One-Class SVM (SGD)  & 9,52\%  & 0,00\% \\
Isolation Forest     & 64,41\% & 5,73\% \\
Local Outlier Factor & 9,52\%  & 0,86\% \\
Robust Covariance    & 66,67\% & 0,75\% \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 6: Insert the three figures**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../results/comparison/rocauc_vs_size.png}
\caption{ROC-AUC u ovisnosti o veličini uzorka. Isolation Forest i One-Class SVM ostaju u pojasu 0,94--0,98; LOF pada k nasumičnom, SGD-OCSVM je invertiran.}
\label{fig:auc}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../results/comparison/runtime_vs_size.png}
\caption{Ukupno vrijeme izvođenja (logaritamska skala). One-Class SVM raste $\approx O(n^2)$, dok Isolation Forest ostaje gotovo ravan.}
\label{fig:runtime}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../results/80k/metrics_bar.png}
\caption{Pregled mjera na poduzorku od 80k za svih sedam modela.}
\label{fig:bars}
\end{figure}
```

- [ ] **Step 7: Verify** — run helper. Expected: `begin == end`; cite `agyemang` `OK`; all three img paths `OK` (`../results/comparison/rocauc_vs_size.png`, `../results/comparison/runtime_vs_size.png`, `../results/80k/metrics_bar.png`).
- [ ] **Step 8: Commit** — `git commit -m "docs: add Results tables (III–VII) and figures (2–4)"`

---

### Task 8: Discussion prose (REZULTATI I RASPRAVA continued)

**Files:**
- Modify: `seminar.tex` (prose after the tables/figures from Task 7)

**Interfaces:**
- Consumes labels `tab:main80k`, `tab:auc`, `tab:f1`, `tab:runtime`, `tab:agyemang`, `fig:auc`, `fig:runtime`, `fig:bars`. Consumes cite `agyemang`.

**Content brief** (Croatian prose, ~5–6 short paragraphs; reference tables/figures by `\ref`):
1. **Glavni nalaz (80k):** uputiti na Tablicu~\ref{tab:main80k}; One-Class SVM (0,9635) i Isolation Forest (0,9612) najbolji rangeri; objasniti da je visoka točnost posljedica neravnoteže, ne kvalitete.
2. **Stabilnost (sweep):** Tablica~\ref{tab:auc} i Slika~\ref{fig:auc} — IF i OCSVM 0,94–0,98 kroz sve veličine; Autoencoder stabilan 0,93–0,97.
3. **Razlozi neuspjeha:** LOF kolabira (0,77→~0,51, nasumično) zbog lokalne gustoće u 30-D; SGD-OCSVM invertiran (ROC-AUC < 0,5) — patologija optimizacije, ne podataka.
4. **Robust Covariance:** dobar ranker (~0,89–0,94) ali MAD označava ~40% (Tablica~\ref{tab:main80k}, stupac impl.\ kontam.), pa F1≈0; R² pada na $\approx-245$.
5. **Jaz rangiranje↔prag (ključna poruka):** Tablica~\ref{tab:f1} — F1 nizak za sve jer label-free MAD prag preflagira naspram 0,17%; R² je posljedično jako negativan i ne razlikuje dobre od loših rangera — to radi ROC-AUC.
6. **Skalabilnost:** Tablica~\ref{tab:runtime} i Slika~\ref{fig:runtime} — OCSVM $\approx O(n^2)$ (16,5→894 s), IF gotovo ravan (0,3→1,7 s); više podataka kupuje OCSVM-u vrijeme, ne kvalitetu. Napomenuti **CPU/GPU caveat** za Autoencoder (jedini na GPU; runtime nije usporediv 1:1, ROC-AUC jest).
7. **Usporedba s referentnim radom:** Tablica~\ref{tab:agyemang} i `\cite{agyemang}` — F1 pada prelaskom s 220 2-D točaka (10% outliera) na 30-D realne podatke (0,17%); LOF i SGD podbacuju u obje studije; ovo izravno odgovara na otvoreno pitanje rada o validaciji na stvarnim podacima.

- [ ] **Step 1:** Write the discussion prose per the brief.
- [ ] **Step 2: Verify** — run helper; balance OK, cite `agyemang` `OK`.
- [ ] **Step 3: Commit** — `git commit -m "docs: write Results discussion"`

---

### Task 9: Abstract + Conclusion (ZAKLJUČAK)

**Files:**
- Modify: `seminar.tex` (`% TASK 9` in abstract and in conclusion)

**Interfaces:**
- Conclusion may cite `agyemang`.

**Content brief:**
- **Abstract** (≤150 riječi, Croatian): motivacija (prijevare, 0,17% neravnoteža) → cilj (validirati Agyemang na velikom realnom skupu; usporediti 6 nenadziranih + nadzirani baseline) → metode (label-free MAD prag, ROC-AUC primarna, sweep 30k–230k) → rezultati (IF/OCSVM 0,94–0,98 i stabilni; jaz rangiranje↔prag; IF najbolji omjer kvaliteta/brzina). Provjeriti broj riječi ≤150.
- **Conclusion** (Croatian, ~1–2 paragrafa): ponoviti cilj; glavni nalaz (label-free detekcija; ROC-AUC pokazuje IF/OCSVM kao izvrsne i stabilne; IF najbolji omjer kvaliteta/brzina); ograničenja (varijanca poduzorka; MAD preflagira; R² nije diskriminativan; SGD-OCSVM degenerira); budući rad (kalibracija praga bez oznaka = otvoreni problem).

- [ ] **Step 1:** Write the abstract inside `\begin{abstract}...\end{abstract}`.
- [ ] **Step 2:** Write the conclusion under ZAKLJUČAK.
- [ ] **Step 3: Verify word count** — `Run: awk 'f&&/\\end\{abstract\}/{f=0} f{print} /\\begin\{abstract\}/{f=1}' seminar.tex | wc -w` → expected ≤ ~150.
- [ ] **Step 4: Verify** — run helper; balance OK.
- [ ] **Step 5: Commit** — `git commit -m "docs: write Abstract and Conclusion"`

---

### Task 10: Final structural lint + compile instructions

**Files:**
- Modify: `seminar.tex` only if the lint finds issues.

- [ ] **Step 1: Full structural lint** — run the Verification helper. Required: `begin == end`; **every** cite `OK` (`agyemang`, `kaggle`, `sklearn`, `ocsvm`, `iforest`, `lof`, `mcd`, `mad`, `pytorch`); **every** img `OK` (3 paths). Also run:
  ```bash
  grep -c '\\section' seminar.tex   # expect 5 (UVOD, METODOLOGIJA, STUDIJA SLUČAJA, REZULTATI I RASPRAVA, ZAKLJUČAK)
  grep -c '\\subsection' seminar.tex # expect 2 (Skup podataka, Vrednovanje)
  grep -oE 'label\{[^}]+\}' seminar.tex | sort   # confirm every \ref target exists
  ```
  Cross-check that every `\ref{...}` has a matching `\label{...}`:
  ```bash
  for r in $(grep -oE '\\ref\{[^}]+\}' seminar.tex | sed -E 's/\\ref\{|\}//g' | sort -u); do
    grep -q "label{$r}" seminar.tex && echo "OK ref $r" || echo "MISSING label $r"; done
  ```
  Fix any `MISSING` inline.
- [ ] **Step 2: Confirm `main.tex` untouched and `ieeeconf.cls` present beside `seminar.tex`**:
  ```bash
  git status --porcelain "Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/main.tex"  # expect empty
  ls "Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/ieeeconf.cls"                  # exists
  ```
- [ ] **Step 3: Provide compile instructions to the user** (no local toolchain): the user compiles with pdfLaTeX twice (so the bibliography/refs resolve), either on Overleaf (upload the whole `Preparation_of_Papers_...` folder so `ieeeconf.cls` and the relative `../results/...` paths resolve — note: on Overleaf the `results/` images must also be uploaded, preserving the `../results/...` relative path, or the includegraphics paths adjusted) or locally via MiKTeX: `pdflatex seminar.tex` ×2. Flag that figure floats / TikZ placement may need minor tuning after the first compile.
- [ ] **Step 4: Commit any lint fixes** — `git commit -m "docs: final structural lint pass on seminar"` (skip if nothing changed).

---

## Self-Review

**1. Spec coverage:**
- Build/preamble/Croatian packages → Task 1. ✓
- Title + placeholder author → Task 1. ✓
- Abstract ≤150 words → Task 9. ✓
- Introduction (Agyemang gap, refs) → Task 3. ✓
- Methodology (lineup+hyperparams Table I, unified score, MAD eq, metrics, AE, CPU/GPU caveat, TikZ Fig.1) → Tasks 4 & 5 (caveat also reinforced in Task 8 discussion). ✓
- Case Study (Datasets + Table II, Evaluation) → Task 6. ✓
- Results (Table III + Tables IV–VI + Table VII + Figs 2–4 + discussion + ranking↔threshold gap + scalability + Agyemang) → Tasks 7 & 8. ✓
- Conclusion → Task 9. ✓
- 9 references → Task 2. ✓
- Acceptance: structure matches `main.tex`, numbers trace to `results/`, ROC-AUC framing, captions IEEE-style, all cites used → covered across tasks + Task 10 lint. ✓
- "Compiles to PDF, no undefined refs/citations": cannot run locally (no toolchain); approximated by the structural lint (Task 10) + user compile. Documented as a constraint. ✓

**2. Placeholder scan:** The only intentional placeholder is the author block (a user decision per spec). `% TASK N` markers are scaffolding removed during execution, not deliverable placeholders. All tables/equation/TikZ/bibliography contain exact content. No "TBD/add error handling/etc." ✓

**3. Type/label consistency:** Labels defined and referenced consistently — `tab:lineup`, `tab:dataset`, `tab:main80k`, `tab:auc`, `tab:f1`, `tab:runtime`, `tab:agyemang`, `fig:pipeline`, `fig:auc`, `fig:runtime`, `fig:bars`, `eq:mad`. Cite keys defined in Task 2 and consumed in Tasks 3,4,6,7,8,9 match exactly. ✓
