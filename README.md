# 🧠 Aplicativo de Diagnóstico de Câncer de Mama

Aplicativo web interativo para predição de câncer de mama utilizando Machine Learning. O sistema utiliza um modelo de Regressão Logística para classificar nódulos mamários como **benignos** ou **malignos** com base em características extraídas de imagens de células.

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido com o objetivo de criar uma interface interativa para predição de câncer de mama utilizando técnicas de Machine Learning. O aplicativo permite que usuários insiram características de um nódulo mamário através de sliders e recebam uma predição sobre se o nódulo é benigno ou maligno, juntamente com as probabilidades associadas.

### Objetivos

- ✅ Limpar e processar dados de câncer de mama
- ✅ Treinar modelo de Regressão Logística para predição
- ✅ Alcançar acurácia de 85% ou superior
- ✅ Construir painel interativo com Streamlit
- ✅ Implementar funcionalidades educacionais para entender o funcionamento interno dos algoritmos

## ✨ Características

- 🎛️ **Interface Interativa**: Sliders para ajuste de 30 características do nódulo
- 📊 **Visualização em Tempo Real**: Gráfico radar mostrando as características normalizadas
- 🔄 **Botão de Reset**: Resetar todos os valores para médias do dataset
- 📈 **Probabilidades**: Exibição de probabilidades de cada classe
- 📚 **Seção Sobre**: Informações detalhadas sobre o projeto, modelo e dataset
- 🎨 **Interface Moderna**: Design limpo e intuitivo

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**: Linguagem de programação principal
- **Streamlit**: Framework para criação da interface web
- **Scikit-learn**: Biblioteca de Machine Learning
  - LogisticRegression
  - StandardScaler
  - train_test_split
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Plotly**: Visualizações interativas (gráficos radar)
- **Pickle**: Serialização do modelo treinado

## 📁 Estrutura do Projeto

```
app-cancer/
├── app/
│   ├── main.py              # Aplicação principal Streamlit
│   └── functions.py          # Funções auxiliares (sidebar, predições, etc.)
├── data/
│   └── cancer_data.csv       # Dataset de câncer de mama
├── model/
│   ├── model.py             # Funções de treinamento do modelo
│   ├── model.pkl            # Modelo treinado (gerado após treinamento)
│   └── scaler.pkl           # Scaler treinado (gerado após treinamento)
├── processing/
│   └── cleaning.py          # Funções de limpeza e processamento de dados
├── scope/
│   ├── logistic_regression.py  # Implementação de referência
│   ├── scaler.py             # Implementação de referência
│   └── tts.py                # Implementação de referência
├── main.py                   # Script para treinar o modelo
├── requirements.txt          # Dependências do projeto
└── README.md                 # Este arquivo
```

## 📊 Dataset

O modelo foi treinado usando o **Breast Cancer Dataset** do Kaggle:

- **Fonte**: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset)
- **Amostras**: 569 casos de células mamárias
- **Características**: 30 características por amostra
  - 10 características principais medidas em 3 formas:
    - **Médias** (`_mean`): Valores médios
    - **Erro Padrão** (`_se`): Erro padrão dos valores
    - **Piores Valores** (`_worst`): Maiores valores encontrados

### Características Analisadas

1. Raio
2. Textura
3. Perímetro
4. Área
5. Suavidade
6. Compacidade
7. Concavidade
8. Pontos Côncavos
9. Simetria
10. Dimensão Fractal

## 🤖 Modelo de Machine Learning

### Especificações

- **Algoritmo**: Regressão Logística
- **Pré-processamento**: StandardScaler (normalização dos dados)
- **Divisão dos Dados**: 80% treino / 20% teste
- **Random State**: 42 (para reprodutibilidade)
- **Objetivo de Acurácia**: 85% ou superior

### Página de Diagnóstico

- **Sliders Interativos**: Ajuste de 30 características organizadas em 3 categorias
- **Gráfico Radar**: Visualização das características normalizadas em tempo real
- **Predição**: Classificação do nódulo como Benigno ou Maligno
- **Probabilidades**: Exibição das probabilidades de cada classe
- **Botão Reset**: Resetar todos os valores para médias do dataset

### Página Sobre

- Informações sobre o projeto
- Detalhes do dataset utilizado
- Especificações do modelo
- Descrição das características analisadas
- Avisos importantes
- Tecnologias utilizadas

## ⚠️ Aviso Importante

**Este aplicativo é apenas para fins educacionais e de demonstração.**

- ❌ Não substitui o diagnóstico médico profissional
- ❌ Não deve ser usado como única fonte de informação para decisões médicas
- ✅ Sempre consulte um médico qualificado para diagnóstico e tratamento
- ⚠️ Os resultados são baseados em um modelo estatístico e podem conter erros

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto é de código aberto e está disponível para fins educacionais.

## 👤 Autor

Desenvolvido como projeto de aprendizado em Machine Learning.

---

**Nota**: Este projeto foi criado com o objetivo de aprender e demonstrar conceitos de Machine Learning aplicados a problemas de saúde. Não deve ser usado para diagnósticos médicos reais.
