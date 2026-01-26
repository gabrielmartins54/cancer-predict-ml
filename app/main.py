import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from functions import clean_data, make_sidebar, predictions, show_about_section

def radar_chart(slider_val, data):
    colors = ['#1f77b4', '#ffbb00', '#d62728']
    fig = go.Figure()

    categories_pt = [
        'Raio', 'Textura', 'Perímetro',
        'Área', 'Suavidade', 'Compacidade',
        'Concavidade', 'Pontos Côncavos',
        'Simetria','Dimensão Fractal'
        ]

    categories = [
        'radius', 'texture', 'perimeter',
        'area', 'smoothness', 'compactness',
        'concavity', 'concave points',
        'symmetry', 'fractal_dimension'
    ]

    labels = ['Média', 'Erro Padrão', 'Piores Valores']
    suffixes = ['_mean', '_se', '_worst']


    for i, (label, suffix) in enumerate(zip(labels, suffixes)):
        values = []
        for category in categories:
            column_name = category.lower() + suffix
            current_val = slider_val.get(column_name, 0)

            min_val = float(data[column_name].min())
            max_val = float(data[column_name].max())
            if (max_val - min_val) > 0:
                norm_val = (current_val - min_val) / (max_val - min_val)
            else:
                norm_val = 0
            values.append(norm_val)

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_pt,
            fill='toself',
            name=label,
            line=dict(color=colors[i], width=2),
            fillcolor=colors[i],
            opacity=0.8,
            hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
        ))

    fig.update_layout(
        template='plotly_dark',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, showline=False),
            angularaxis=dict(direction='clockwise')
        ),
        height=500,
        margin=dict(l=80, r=80, t=20, b=20),
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(
        page_title='Diagnóstico de Câncer de Mama',
        page_icon=':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
        )

    # Tabs para navegação
    tab1, tab2 = st.tabs(['🏠 Diagnóstico', '📚 Sobre'])
    
    with tab1:
        input_data = clean_data()
        slider_values = make_sidebar()

        with st.container():
            st.title('Diagnóstico de Câncer de Mama')
            st.write('Esse aplicativo prevê se um tumor é benigno ou maligno baseado nos parâmetros informados')

        col1, col2 = st.columns([4,1])

        with col1:
            chart = radar_chart(slider_values, input_data)
            st.plotly_chart(chart, use_container_width=True)
        with col2:
            predictions(slider_values)
    
    with tab2:
        show_about_section()

if __name__ == '__main__':
    main()