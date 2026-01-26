import pandas as pd
import streamlit as st
import pickle
import numpy as np

def clean_data():
    '''
    Returns
        data : -> pandas DataFrame
    '''
    #read the data
    data = pd.read_csv('data/cancer_data.csv')

    #cleaning the data
    data = data.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)
    
    return data

def sliders_data():
    slider_sections = {
    'Médias': [
        ('Raio', 'radius_mean'),
        ('Textura', 'texture_mean'),
        ('Perímetro', 'perimeter_mean'),
        ('Área', 'area_mean'),
        ('Suavidade', 'smoothness_mean'),
        ('Compacidade', 'compactness_mean'),
        ('Concavidade', 'concavity_mean'),
        ('Pontos Côncavos', 'concave points_mean'),
        ('Simetria', 'symmetry_mean'),
        ('Dimensão Fractal', 'fractal_dimension_mean'),
    ],

    'Erro Padrão': [
        ('Raio', 'radius_se'),
        ('Textura', 'texture_se'),
        ('Perímetro', 'perimeter_se'),
        ('Área', 'area_se'),
        ('Suavidade', 'smoothness_se'),
        ('Compacidade', 'compactness_se'),
        ('Concavidade', 'concavity_se'),
        ('Pontos Côncavos', 'concave points_se'),
        ('Simetria', 'symmetry_se'),
        ('Dimensão Fractal', 'fractal_dimension_se'),
    ],

    'Piores Valores': [
        ('Raio', 'radius_worst'),
        ('Textura', 'texture_worst'),
        ('Perímetro', 'perimeter_worst'),
        ('Área', 'area_worst'),
        ('Suavidade', 'smoothness_worst'),
        ('Compacidade', 'compactness_worst'),
        ('Concavidade', 'concavity_worst'),
        ('Pontos Côncavos', 'concave points_worst'),
        ('Simetria', 'symmetry_worst'),
        ('Dimensão Fractal', 'fractal_dimension_worst'),
    ]
    }

    return slider_sections 

def get_mean_values(data):
    '''
    Retorna um dicionário com os valores médios de todas as características.
    
    Args:
        data : pandas DataFrame
    
    Returns:
        dict : Dicionário com valores médios
    '''
    mean_values = {}
    sliders = sliders_data()
    
    for key, list in sliders.items():
        for tuple in list:
            mean_values[tuple[1]] = float(data[tuple[1]].mean())
    
    return mean_values

def make_sidebar(sliders=sliders_data(), data=clean_data()):
    '''
    Args:
        sliders : list of a list of tuples
        data : pandas DataFrame
    '''
    st.sidebar.header('Características do Nódulo')
    
    # Botão de reset
    if st.sidebar.button('🔄 Resetar Valores', use_container_width=True, help='Resetar todos os sliders para valores médios'):
        mean_values = get_mean_values(data)
        for key in mean_values:
            st.session_state[key] = mean_values[key]
        st.rerun()
    
    st.sidebar.divider()
    
    input_dict = {}

    for key, list in sliders.items():
        with st.sidebar.expander(label=key):
            for tuple in list:
                input_dict[tuple[1]] = st.slider(
                    label=tuple[0],
                    min_value=float(data[tuple[1]].min()),
                    max_value=float(data[tuple[1]].max()),
                    value=float(data[tuple[1]].mean()),
                    key=tuple[1]
                    )
    return input_dict   

def predictions(data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    pred_array = np.array(list(data.values())).reshape(1, -1)

    scale_array = scaler.transform(pred_array)
    prediction = model.predict(scale_array)

    st.subheader('Resultado')
    st.write('O nódulo é:')

    if prediction == 0:
        st.badge(label='Benigno', color='green', icon='🟢', width='stretch')
    else:
        st.badge(label='Maligno', color='red', icon='🔴', width='stretch')
    
    st.write("Probabilidade de ser benigno: ", model.predict_proba(scale_array)[0][0].round(2))
    st.write("Probabilidade de ser maligno: ", model.predict_proba(scale_array)[0][1].round(2))
    

def show_about_section():
    st.header('📚 Sobre o Projeto')
    
    st.markdown("""
    ### 🎯 Objetivo
    Este aplicativo utiliza um modelo de **Regressão Logística** para prever se um nódulo mamário é 
    **benigno** ou **maligno** com base em características extraídas de imagens de células.
    """)
    
    st.markdown("""
    ### 📊 Dataset
    O modelo foi treinado usando o **Breast Cancer Dataset** do Kaggle, que contém:
    - **569 amostras** de células mamárias
    - **30 características** por amostra, incluindo:
      - **Médias**: Valores médios das características das células
      - **Erro Padrão**: Erro padrão dos valores
      - **Piores Valores**: Maiores valores encontrados (mais preocupantes)
    
    **Fonte do Dataset:** [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset)
    """)
    
    st.markdown("""
    ### 🤖 Modelo de Machine Learning
    - **Algoritmo**: Regressão Logística
    - **Pré-processamento**: StandardScaler (normalização dos dados)
    - **Divisão dos dados**: 80% treino / 20% teste
    - **Objetivo de Acurácia**: 85% ou superior
    
    O modelo analisa as características do nódulo e retorna:
    - **Classificação**: Benigno ou Maligno
    - **Probabilidades**: Probabilidade de cada classe
    """)
    
    st.markdown("""
    ### 📋 Características Analisadas
    O modelo utiliza 10 características principais, cada uma medida em três formas:
    1. **Raio**: Tamanho médio das distâncias do centro aos pontos do perímetro
    2. **Textura**: Desvio padrão dos valores da escala de cinza
    3. **Perímetro**: Tamanho do perímetro da célula
    4. **Área**: Área da célula
    5. **Suavidade**: Variação local no comprimento do raio
    6. **Compacidade**: Perímetro² / área - 1.0
    7. **Concavidade**: Severidade das porções côncavas do contorno
    8. **Pontos Côncavos**: Número de porções côncavas do contorno
    9. **Simetria**: Simetria da célula
    10. **Dimensão Fractal**: Aproximação "coastline" - 1.0
    """)
    
    st.markdown("""
    ### ⚠️ Aviso Importante
    **Este aplicativo é apenas para fins educacionais e de demonstração.**
    
    - Não substitui o diagnóstico médico profissional
    - Não deve ser usado como única fonte de informação para decisões médicas
    - Sempre consulte um médico qualificado para diagnóstico e tratamento
    - Os resultados são baseados em um modelo estatístico e podem conter erros
    """)
    
    st.markdown("""
    ### 🛠️ Tecnologias Utilizadas
    - **Python**: Linguagem de programação
    - **Streamlit**: Framework para interface web
    - **Scikit-learn**: Biblioteca de machine learning
    - **Pandas**: Manipulação e análise de dados
    - **Plotly**: Visualizações interativas
    - **NumPy**: Computação numérica
    """)
    
    st.markdown("""
    ### 📁 Estrutura do Projeto
    ```
    app-cancer/
    ├── app/              # Aplicação Streamlit
    ├── data/             # Dataset
    ├── model/            # Modelo treinado
    ├── processing/       # Processamento de dados
    └── scope/            # Implementações de referência
    ```
    """)