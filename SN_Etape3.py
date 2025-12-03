#-----------------------------------------------------------
# Python for Data Science : SN
# Etape 3 : Application Streamlit   
#-----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Importation du Dataset
df = pd.read_csv('plant_disease_dataset.csv')
data = pd.read_csv('dataset_machine_learning.csv')
model_xgb = pickle.load(open('plant_disease_model_xgb.sav', 'rb'))

st.sidebar.image('Photos\images2.jpg',width=200)

def main():
    st.markdown("<h1 style='text-align:center;color: green;'>PLANT DISEASES APP</h1>",unsafe_allow_html=True)
    menu = ['Accueil', 'Analyses', 'Visualisation', 'Machine Learning']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice=='Accueil':
        st.header('A propos des maladies des plantes ?')
        st.image('Photos\disease.jpg')
        st.write("Chez les plantes, la maladie est un état phénotypique anormal, qui présente des écarts, appelés « symptômes », par rapport au phénotype normal attendu, et qui réduit la croissance de la plante, sa productivité et son utilité pour l'homme.")
        st.write("Les maladies des plantes sont des problèmes courants en agriculture et jardinage, car elles peuvent affecter la croissance, le rendement et la santé des végétaux. Parmi les maladies les plus fréquentes, on trouve la brûlure (blight), la rouille (rust) et la moisissure (mildew).")
        st.write("Voici une brève description de chacune de ces maladies :\n")
        st.write("\n\n**La moisissure (mildew)**")
        st.image('Photos\mildew.jpg', caption='Moisissure (Mildew)')
        st.write("\nLa moisissure est une maladie fongique qui affecte principalement les feuilles des plantes. Elle se manifeste par une poudre blanche ou grise sur la surface des feuilles, qui peut entraîner le jaunissement, le flétrissement et la chute prématurée des feuilles.")
        st.write("\n\n**La brûlure (blight)**")
        st.image('Photos\Blight.jpg', caption='Brûlure (Blight)')
        st.write("\nLa brûlure est une maladie fongique qui provoque le flétrissement et la mort rapide des feuilles, des tiges et des fruits. Elle se manifeste par des taches brunes ou noires sur les feuilles, qui s'étendent rapidement et peuvent entraîner la chute prématurée des feuilles.\n")
        st.write("\n\n**La rouille (rust)**")
        st.image('Photos\Rust.jpg', caption='Rouille (Rust)')
        st.write("\nLa rouille est une maladie fongique qui affecte principalement les feuilles des plantes. Elle se manifeste par des pustules orange, jaunes ou brunes sur la surface des feuilles, qui peuvent entraîner le jaunissement, le flétrissement et la chute prématurée des feuilles.")

    elif choice=='Analyses':
        st.header('Analyses des données')
        st.subheader('Plant diseaese dataset')
        st.write(df.head(10))

        if st.checkbox('Summary'):
            st.write(df.describe())
        elif st.checkbox('Correlation'):
            fig = plt.figure(figsize=(15,15))
            st.write(sns.heatmap(data.corr(),annot=True))
            st.pyplot(fig)

    elif choice=='Visualisation':
        st.header('Visualisation des données')
        st.subheader('Plant disease dataset')
        st.write(df.head())

        if st.checkbox('Distribution des types de maladies'):
            fig = plt.figure(figsize=(6,4))
            sns.countplot(x='disease_type', hue='pesticide', data=df)
            plt.title("Répartition des types de maladies")
            st.pyplot(fig)
        elif st.checkbox('Boxplot des variables numériques par type de maladie'):
            for col in ['leaf_length', 'leaf_width', 'stem_diameter']:
                fig = plt.figure(figsize=(6,4))
                sns.boxplot(x='disease_type', y=col, data=data)
                plt.title(f"{col} selon le type de maladie")
                st.pyplot(fig)
        elif st.checkbox('Distribution des types de sol et météo'):
            fig, axs = plt.subplots(1, 2, figsize=(12,4))
            sns.countplot(x='soil_type', hue='disease_type', data=df, ax=axs[0])
            sns.countplot(x='weather', hue='disease_type', data=df, ax=axs[1])
            axs[0].set_title("Répartition des types de sol")
            axs[1].set_title("Répartition des types de météo")
            st.pyplot(fig)

    elif choice=='Machine Learning':
        st.subheader('Machine Learning')
        action = st.selectbox("Choisir une action", ['A propos du modèle', 'Prediction'])
        if action == 'A propos du modèle':
            st.write("Modèle de classification des maladies des plantes utilisant XGBoost")
            st.write("Le modèle XGBoost a été choisi pour sa performance et sa capacité à gérer des données complexes. Il a été entraîné sur un ensemble de données comprenant diverses caractéristiques des plantes et leurs maladies associées.")
            X = data.drop('disease_type', axis=1)
            y = data['disease_type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model_xgb.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.markdown("### Performance du modèle")
            st.write(f"**Accuracy :** {acc:.2f}")
            st.write("**Matrice de confusion :**")
            fig= plt.figure(figsize=(6,4))
            sns.heatmap(cm,annot=True)
            plt.xlabel('Prédictions')
            plt.ylabel('Véritables valeurs')
            st.pyplot(fig)
            st.write("**Rapport de classification :**")
            st.json(report)

        elif action == 'Prediction':
            st.markdown("### Prédiction d'une maladie à partir de vos paramètres")
            # --- Saisie utilisateur ---
            leaf_length = st.number_input("Longueur de la feuille", min_value=0.0, max_value=30.0, value=10.0)
            leaf_width = st.number_input("Largeur de la feuille", min_value=0.0, max_value=20.0, value=5.0)
            stem_diameter = st.number_input("Diamètre de la tige", min_value=0.0, max_value=10.0, value=1.0)
            pesticide = st.selectbox("Pesticide utilisé", ['Oui', 'Non'])
            if pesticide == 'Oui':
                pesticide_enc = 1
            else:
                pesticide_enc = 0
            soil_type = st.selectbox("Type de sol", sorted(df['soil_type'].unique()))
            if soil_type == 'loamy':
                soil_type_enc = 0.336
            else:
                soil_type_enc = 0.332
            weather = st.selectbox("Type de météo", sorted(df['weather'].unique()))
            if weather == 'sunny':
                weather_enc = 0.352
            elif weather == 'rainy':
                weather_enc = 0.346
            else:
                weather_enc = 0.302

            if st.button("Prédire le type de maladie"):
                input_data = np.array([[leaf_length, leaf_width, stem_diameter, pesticide_enc, soil_type_enc, weather_enc]])
                pred = model_xgb.predict(input_data)[0]
                proba = model_xgb.predict_proba(input_data).max()
                if pred == 0:
                    st.success(f"Type de maladie prédit : blight")
                elif pred == 1:
                    st.success(f"Type de maladie prédit : mildew")
                else:
                    st.success(f"Type de maladie prédit : rust")
                st.info(f"Probabilité associée : {(proba)*100:.2f}%")



if __name__== '__main__':

    main()
