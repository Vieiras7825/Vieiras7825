import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dados
dados = {
    'address': ['Rua Herval', 'Avenida São Miguel', 'Rua Oscar Freire', 'Rua Júlio Sayago', 'Rua Barata Ribeiro', 'Rua Domingos Paiva', 'Rua Guararapes'],
    'district': ['Belenzinho', 'Vila Marieta', 'Pinheiros', 'Vila Ré', 'Bela Vista', 'Brás', 'Brooklin Paulista'],
    'area': [21, 15, 18, 56, 19, 50, 72],
    'bedrooms': [1, 1, 1, 2, 1, 2, 2],
    'garage': [0, 1, 0, 2, 0, 1, 1],
    'type': ['Studio e kitnet', 'Studio e kitnet', 'Apartamento', 'Casa em condomínio', 'Studio e kitnet', 'Apartamento', 'Apartamento'],
    'rent': [2400, 1030, 4000, 1750, 4000, 3800, 3500],
    'total': [2939, 1345, 4661, 1954, 4654, 4587, 5187]
}

# Importação dos dados para um DataFrame
df = pd.DataFrame(dados)

# Verificação de valores nulos
print(df.isnull().sum())

# Renomeando colunas
df.rename(columns={'address': 'Endereço', 'district': 'Bairro', 'area': 'Área', 'bedrooms': 'Quartos', 'garage': 'Garagem', 'type': 'Tipo', 'rent': 'Aluguel', 'total': 'Total'}, inplace=True)

# Análise Descritiva
print(df.describe())

# Verificação de duplicatas
print('Duplicatas:', df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Conversão de tipos de dados
df['Área'] = df['Área'].astype(float)

# Codificação de variáveis categóricas
df = pd.get_dummies(df, columns=['Tipo'], drop_first=True)

# Modelagem de Regressão Linear
X = df[['Área', 'Quartos', 'Garagem']]  # Variáveis independentes
y = df['Aluguel']  # Variável dependente

# Ajuste do modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Previsões
previsoes = modelo.predict(X)

# Avaliação do modelo
print('R²:', r2_score(y, previsoes))
print('Erro médio quadrático:', mean_squared_error(y, previsoes))

# Visualização de Dados
plt.scatter(df['Área'], df['Aluguel'], color='blue')
plt.plot(df['Área'], previsoes, color='red', linewidth=3)
plt.title('Área vs Aluguel')
plt.xlabel('Área')
plt.ylabel('Aluguel')
plt.show()








