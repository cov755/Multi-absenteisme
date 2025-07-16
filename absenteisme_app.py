# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import streamlit as st
import plotly.express as px

# Configuration
st.set_page_config(page_title="Covéa | Analyse du multi-absentéisme", layout="wide")
sns.set(style="whitegrid")

# Titre
col1,col2=st.columns([0.1,0.9])
with col1 :
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fc/Logo_Covéa.png",width=70)
with col2:
    st.title("Analyse du multi-absentéisme - Pilotage")
    
# Téléversement du fichier
uploaded_file = st.file_uploader("Choisissez un fichier Excel (.xlsx)", type="xlsx")

if uploaded_file:
# Lecture du fichier
    df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    df = df.drop(columns=['Sexe (libellé)'], errors='ignore')

# Calcul taux d'absentéisme par direction
    tauxdir = df.groupby('Direction')['Total jrs maladie'].sum() / df.groupby('Direction')['Jours calendaires travaillés'].sum()
    tauxdir = tauxdir.sort_values(ascending=False)
    df['tauxdir'] = df['Direction'].map(tauxdir)

# Filtrer ≥ 3 arrêts
    df_3arrets = df[df['Total Arrêts maladie'] >= 3]
    
    a=df_3arrets['Total Arrêts maladie'].sum()
    b=df['Total Arrêts maladie'].sum()    
    
# Statistiques globales
    st.subheader("📌 Statistiques clés")
    col1, col2, col3,col4 = st.columns(4)
    with col1:
        st.metric("Total arrêts (≥3)", df_3arrets['Total Arrêts maladie'].sum())
        taux_arret_covea = round((a / b) * 100, 1)
        st.metric("Taux arrêt maladie Covéa", f"{taux_arret_covea} %")

        taux_arret = df_3arrets['Total Arrêts maladie'].mean()
        st.metric("Taux d'arrêt par salarié", f"{taux_arret:.1f} %")

    with col2:
        taux_abs = df['Total jrs maladie'].sum() / df['Jours calendaires travaillés'].sum()
        st.metric("Taux absentéisme global", f"{taux_abs:.2%}")
        
        # Calcul du taux d'absentéisme global multi-asbenteisme
        taux_absenteisme_global = df_3arrets['Total jrs maladie'].sum() / df['Jours calendaires travaillés'].sum()
        st.metric("Taux du multi-absentéisme global", f"{taux_absenteisme_global:.2%}")
        
        am=df_3arrets['Total jrs maladie'].sum()/df['Total jrs maladie'].sum()
        st.metric("Taux absences maladie", f"{am:.2%}")
        
        # Calculer le pourcentage des arrêts de 3 jours ou moins
        pourcentage_arrets_courts = (
            df_3arrets['Arrêts 3 jours ou -'].sum() / df_3arrets['Total Arrêts maladie'].sum()
            ) 
        st.metric("Taux arrêts maladie de durée ≤ 3 jours", f"{pourcentage_arrets_courts:.2%}")


    with col3:
        ja=df_3arrets['Total jrs maladie'].sum()
        st.metric("Nombre de jours d'absences", f"{ja:.0f} jours")

        da=df_3arrets['Total jrs maladie'].mean()
        st.metric("Durée moyenne d'absences",f"{da:.1f} jours")
        
        duree_moyenne = df_3arrets['Total jrs maladie'].sum() / df_3arrets['Total Arrêts maladie'].sum()
        st.metric("Durée moyenne d’un arrêt", f"{duree_moyenne:.1f} jours")
    
    with col4:
        df['groupe_arrets'] = df['Total Arrêts maladie'].apply(lambda x: '≥3 arrêts' if x >= 3 else '<3 arrêts')
        counts = df['groupe_arrets'].value_counts()

    # Afficher les métriques par groupe
        st.metric("Nombre de salariés avec ≥3 arrêts", counts.get('≥3 arrêts', 0))
        st.metric("Nombre de salariés avec <3 arrêts", counts.get('<3 arrêts', 0))

    # Filtrer les salariés absents pour maladie
        df_absents = df[df['Salarié absent maladie'] == 1]
        total_absents = len(df_absents)

    # Recalculer le nombre de salariés avec ≥3 arrêts
        nb_3_arrets = len(df[df['Total Arrêts maladie'] >= 3])

    # Calculer le pourcentage
        pourcentage_3plus = (nb_3_arrets / total_absents) * 100 if total_absents > 0 else 0
        st.metric("Taux absents avec ≥3 arrêts", f"{pourcentage_3plus:.1f} %")

        sa = len(df[df['Total Arrêts maladie'] >= 3]) / len(df['Matricule anonyme']) * 100
        st.metric("Taux de salariés Covéa", f"{sa:.1f} %")

# Groupe arrêt
    df['Groupe Arrêts'] = df['Total Arrêts maladie'].apply(lambda x: '≥ 3 arrêts' if x >= 3 else '< 3 arrêts')

    st.subheader("📊 Répartition du nombre de salariés par variables")

# Graphiques démographiques
    variables = {
        'Sexe (libellé).1': None,
        'BS Tranches Age': ['20/24', '25/29', '30/34', '35/39', '40/44', '45/49', '50/54', '55/59', '60/64', '65 ans et +'],
        '[Tranches Ancienneté Groupe]': ['<5', '5/9', '10/14', '15/19', '20/24', '25/29', '30/34', '35/39', '40 ans et +'],
        'BS Statut': None
    }

    cols = st.columns(2)
    i = 0
    for var, order in variables.items():
        if var in df.columns:
            fig = px.histogram(
                df, x=var, color='Groupe Arrêts',
                category_orders={var: order} if order else {},
                barmode='group',
                color_discrete_sequence=['#EAE8AD', '#0E2841'],
                text_auto=True
            )
            fig.update_layout(
                title=f"Répartition par {var}",
                xaxis_title=var, yaxis_title="Nombre de salariés"
            )
            fig.update_traces(textposition='outside')
            cols[i % 2].plotly_chart(fig, use_container_width=True)
            i += 1

    st.subheader("🏢 Directions les plus concernées")

    # Pourcentage d'arrêts
    arrets_par_dir_3plus = df_3arrets.groupby('Direction')['Total Arrêts maladie'].sum()
    arrets_par_dir_all = df.groupby('Direction')['Total Arrêts maladie'].sum()
    pourcent_par_dir = (arrets_par_dir_3plus / arrets_par_dir_all * 100).sort_values(ascending=False).head(5)

    col1, col2 = st.columns(2)

    fig1 = px.bar(
        pourcent_par_dir[::-1],
        x=pourcent_par_dir[::-1].values,
        y=pourcent_par_dir[::-1].index,
        orientation='h',
        text=pourcent_par_dir[::-1].apply(lambda x: f"{x:.1f}%"),
        labels={'x': 'Pourcentage (%)', 'y': 'Direction'},
        color_discrete_sequence=['#0E2841']
    )
    fig1.update_layout(title="Top 5 directions par % des arrêts (≥3)")
    col1.plotly_chart(fig1, use_container_width=True)

    # Top directions en effectif
    top_directions = df_3arrets['Direction'].value_counts().nlargest(5)
    fig2 = px.bar(
        x=top_directions.values,
        y=top_directions.index,
        orientation='h',
        text=top_directions.values,
        labels={'x': 'Nombre de salariés', 'y': 'Direction'},
        color_discrete_sequence=['#EAE8AD']
    )
    fig2.update_layout(title="Top 5 directions par salariés avec ≥3 arrêts")
    col2.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Répartition des absences par durée")

    # Répartition en jours d'absence
    tranches_jours = {
        '≤ 3 jours': df_3arrets['Jrs 3 jours ou -'].sum(),
        '4j - 3 mois': df_3arrets['Jrs 4j-3m'].sum(),
        '> 3 mois': df_3arrets['Jrs >3 mois'].sum()
    }

    # Répartition en nombre d'arrêts
    tranches_arrets = {
        '≤ 3 jours': df_3arrets['Arrêts 3 jours ou -'].sum(),
        '4j - 3 mois': df_3arrets['Arrêts 4 jours-3 mois'].sum(),
        '> 3 mois': df_3arrets['Arrêts > 3 mois'].sum()
    }
    
    arrets_tranches = {
    '3 arrêts': (df_3arrets['Total Arrêts maladie'] == 3).sum(),
    '4 arrêts': (df_3arrets['Total Arrêts maladie'] == 4).sum(),
    '5 arrêts': (df_3arrets['Total Arrêts maladie'] == 5).sum(),
    '6-10 arrêts': ((df_3arrets['Total Arrêts maladie'] >= 6) & (df_3arrets['Total Arrêts maladie'] <= 10)).sum(),
    '> 10 arrêts': (df_3arrets['Total Arrêts maladie'] > 10).sum()
    }

    col3, col4,col5 = st.columns(3)

    fig3 = px.bar(
        x=list(tranches_jours.keys()),
        y=list(tranches_jours.values()),
        text=list(tranches_jours.values()),
        labels={'x': 'Tranche', 'y': "Jours d'absence"},
        color_discrete_sequence=['#EAE8AD']
    )
    fig3.update_traces(textposition='outside')
    fig3.update_layout(title="Jours d'absence par tranche")
    col3.plotly_chart(fig3, use_container_width=True)

    fig4 = px.bar(
        x=list(tranches_arrets.keys()),
        y=list(tranches_arrets.values()),
        text=list(tranches_arrets.values()),
        labels={'x': 'Tranche', 'y': "Nombre d'arrêts"},
        color_discrete_sequence=['#0E2841']
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(title="Nombre d'arrêts par tranche")
    col4.plotly_chart(fig4, use_container_width=True)

    fig5 = px.bar(
        x=list(arrets_tranches.keys()),
        y=list(arrets_tranches.values()),
        text=list(arrets_tranches.values()),
        labels={'x': 'Nombre arrêts', 'y': "Nombre de salariés"},
        color_discrete_sequence=['#EAE8AD']
    )
    fig5.update_traces(textposition='outside')
    fig5.update_layout(title="Répartition des salariés par nombre d’arrêts")
    col5.plotly_chart(fig5, use_container_width=True)


    st.success("Analyse terminée")

else:
    st.info("Aucun fichier Excel téléversé")
    
