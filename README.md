# 🩻 Projeto X-rays

**Status do Projeto:** 🚀 MVP Funcional (v1.0)

O **Projeto X-rays** é um projeto de **Data Science e Machine Learning aplicado à saúde**, que utiliza **Visão Computacional** e **Deep Learning** para análise de **radiografias de tórax**, com foco na identificação de padrões associados à **COVID-19**.

Mais do que classificar imagens, o projeto busca **priorizar exames com maior risco clínico**, combinando **confiança do modelo** e **probabilidade estimada de severidade**. O objetivo é apoiar **triagem médica**, **pesquisa científica** e **demonstrações técnicas de IA aplicada à saúde**.

> ⚠️ **Aviso Importante:** Este projeto tem finalidade **educacional e experimental** e **não substitui diagnóstico médico**.

---

## 🧠 Motivação

Radiografias de tórax são exames:

* Rápidos
* De baixo custo
* Amplamente disponíveis

Em cenários de alta demanda, modelos de Deep Learning podem **auxiliar na triagem inicial**, priorizando exames suspeitos para revisão humana. Este projeto segue boas práticas de **Data Science, Machine Learning e MLOps**, com foco em **reprodutibilidade, versionamento de experimentos, modularidade de código e clareza para colaboração em times de dados**.

---

## 📊 Fonte dos Dados

O projeto utiliza o dataset **COVIDx CXR-4**, amplamente adotado em pesquisas acadêmicas.

* **Dataset:** COVIDx CXR-4
* **Origem:** Kaggle (COVIDx CXR)
* **Conteúdo:**

  * Radiografias de tórax (CXR)
  * Classes:

    * COVID-19
    * Pneumonia (não-COVID)
    * Normal

O dataset consolida imagens provenientes de múltiplas fontes públicas, aumentando a **diversidade clínica** e a **robustez estatística** do modelo.

---

## 🏗️ Arquitetura do Projeto (Visão Data/ML)

O **Projeto X-rays** adota uma arquitetura **modular e escalável**, separando claramente:

### 🔧 Backend — Pipeline de Machine Learning (Offline)

* Ingestão e padronização de imagens
* Data augmentation
* Treinamento de modelos Deep Learning
* Avaliação e persistência de métricas e pesos

### 🖥️ Frontend — Visualização e Análise de Resultados

* Dashboard interativo
* Inspeção visual das imagens
* Análise de erros, probabilidades e confiança do modelo

---

## 🛠️ Stack Tecnológica

* **Linguagem:** Python 3.12
* **Manipulação de Dados:** Pandas, NumPy
* **Visão Computacional:** OpenCV, Pillow
* **Deep Learning:** TensorFlow / Keras
* **Arquiteturas:** DenseNet-121
* **Visualização:** Matplotlib, Seaborn, Plotly
* **Dashboard:** Streamlit

---

## 🔄 Pipeline de Data Science & Machine Learning

### 1️⃣ Ingestão e Padronização das Imagens

* Consolidação de imagens de múltiplas fontes
* Redimensionamento padronizado
* Conversão para escala de cinza
* Normalização de pixels
* Validação de integridade (imagens corrompidas/inexistentes)
* Análise e mitigação de desbalanceamento de classes

---

### 2️⃣ Engenharia de Features (Visão Computacional)

* **Data Augmentation**

  * Rotação controlada
  * Flip horizontal
  * Zoom leve
* **Extração Automática de Features**

  * Uso de camadas convolucionais pré-treinadas
* **Embeddings Profundos**

  * Representações vetoriais para análises posteriores

Essas técnicas reduzem **overfitting** e aumentam a capacidade de **generalização** do modelo.

---

### 3️⃣ Modelagem — Machine Learning & Deep Learning

* **Modelo Base:** CNNs pré-treinadas (Transfer Learning)
* **Fine-Tuning:** Ajuste das camadas finais ao contexto radiológico
* **Classificação Multiclasse:**

  * COVID-19
  * Pneumonia
  * Normal

**Boas práticas de ML aplicadas:**

* Early Stopping
* Regularização (Dropout / L2)
* Monitoramento de métricas clínicas relevantes

---

### 4️⃣ Priorização Clínica

Além da classe prevista, o sistema calcula um **Clinical Priority Score**:

```
Prioridade Clínica =
(0.6 × Confiança do Modelo) +
(0.4 × Probabilidade de Severidade)
```

Esse score permite **ordenar exames por risco potencial**, apoiando a triagem e a revisão humana.

---

## 📈 Métricas de Avaliação (ML aplicado à Saúde)

O modelo é avaliado com métricas adequadas ao contexto clínico:

* Accuracy
* Precision
* Recall (Sensibilidade)
* F1-Score
* AUC-ROC
* Matriz de Confusão

Essas métricas ajudam a equilibrar **falsos positivos** e **falsos negativos**, fundamentais em aplicações de saúde.

---

## ▶️ Como Executar o Projeto

### 1️⃣ Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Treinamento do Modelo

```bash
python train_model.py
```

### 3️⃣ Execução do Dashboard

```bash
streamlit run app.py
```

---

## ⚠️ Limitações Técnicas e Considerações de ML

* Uso **educacional e experimental**
* Não substitui diagnóstico médico
* Possível viés devido à heterogeneidade das imagens
* Generalização limitada fora do dataset original
* CNNs requerem técnicas adicionais de interpretabilidade (ex.: Grad-CAM)

---

## 🗺️ Roadmap

### ✔️ Concluído

* EDA visual das imagens
* Pipeline de pré-processamento
* Modelos CNN com Transfer Learning
* Avaliação com métricas clínicas
* Dashboard interativo (v1)

### 🔮 Próximos Passos (Evolução Técnica)

* Grad-CAM para explicabilidade visual
* Ensemble de CNNs
* Classificação de severidade (leve / moderada / grave)
* Integração com dados clínicos estruturados
* Versionamento de modelos e métricas (MLOps)

---

## 🤝 Colaboração, Ownership e Contribuições

Projetos em grupo evoluem em ritmos diferentes. Para evitar que o portfólio fique desatualizado, este repositório adota práticas claras de reconhecimento e continuidade.

### 👥 Equipe do Projeto

* **Alexandre Otsuka** — GitHub: [https://github.com/arotsuka](https://github.com/arotsuka)
* **Elias Yuri Maximo** — GitHub: [https://github.com/Elias-Yuri-Maximo](https://github.com/Elias-Yuri-Maximo)
* **Aurélien Jacomy** — GitHub: [https://github.com/aurelien-jacomy](https://github.com/aurelien-jacomy)
* **William Endo Freire** — GitHub: [https://github.com/wefreire](https://github.com/wefreire)

### 🧩 Principais Contribuições Técnicas

**EDA (Exploratory Data Analysis)**

* Padronização e validação dos metadados
* Análise da distribuição das classes
* Verificação de imagens corrompidas ou inexistentes
* Avaliação da origem das imagens e número por paciente
* Análise estatística dos tamanhos das imagens

**Segmentação (U-Net / U-Net++)**

* Arquitetura encoder–decoder para segmentação pixel a pixel
* Extração hierárquica de features
* Skip connections para preservação espacial
* Saída Sigmoid para mapas de probabilidade binária

**Classificação (CNNs e Transfer Learning)**

* CNN customizada para classificação
* EfficientNet-B0 com pipeline `tf.data`
* Treinamento em duas fases (backbone congelado + fine-tuning)
* Regularização com Dropout
* Saídas probabilísticas e métrica AUC para contexto médico

---

## 📄 Licença e Uso

Este projeto é disponibilizado para fins **educacionais e de pesquisa**. Consulte o arquivo `LICENSE` para mais detalhes.
