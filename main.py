import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# LEITURA DOS DADOS
# ==============================
df = pd.read_csv("dados.csv")

print("Tabela completa:")
print(df)

# ==============================
# ENTRADAS E RESPOSTAS
# ==============================
X = df["texto"]
y = df["categoria"]

print("\nX (entradas):")
print(X)

print("\ny (respostas corretas):")
print(y)

# ==============================
# CRIAÇÃO DAS FERRAMENTAS
# ==============================
vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000)

# ==============================
# TRANSFORMAR TEXTO EM NÚMEROS
# ==============================
X_transformado = vectorizer.fit_transform(X)

print("\nFormato dos dados transformados:")
print(X_transformado.shape)

# ==============================
# TREINAMENTO
# ==============================
model.fit(X_transformado, y)

print("\nModelo treinado com sucesso!")

# ==============================
# TESTES INTERATIVOS
# ==============================
while True:
    entrada = input("\nDigite o que o cliente procurou (ou 'sair'): ")

    if entrada.lower() == "sair":
        print("Encerrando testes da IA.")
        break

    entrada_transformada = vectorizer.transform([entrada])
    resultado = model.predict(entrada_transformada)

    probabilidades = model.predict_proba(entrada_transformada)[0]
    categorias = model.classes_

    print("Categoria prevista:", resultado[0])
    print("Confiança por categoria:")

    for categoria, prob in zip(categorias, probabilidades):
        print(f"- {categoria}: {prob:.2f}")
