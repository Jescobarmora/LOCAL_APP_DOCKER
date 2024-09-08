import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.creation import MathFeatures
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo entrenado
with open('lightgbm_streamlit_model.pkl', 'rb') as file:
    modelo = pickle.load(file)
    
label_mapping = {0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'}

# Función para aplicar la misma ingeniería de variables que en `process_data`
def aplicar_ingenieria_de_variables(base_modelo, cuantitativas_bm, categóricas_bm):
    # Asegurarse de que las variables categóricas están en el tipo de datos correcto
    base_modelo[categóricas_bm] = base_modelo[categóricas_bm].astype('object')

    # Codificación de variables categóricas usando LabelEncoder
    for col in categóricas_bm:
        le = LabelEncoder()
        base_modelo[col] = le.fit_transform(base_modelo[col].astype(str))
        
    # Creación de características matemáticas
    math_transformer = MathFeatures(variables=cuantitativas_bm, func=['sum', 'prod'])
    base_modelo = math_transformer.fit_transform(base_modelo)

    # Creación de características polinómicas
    poly_transformer = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    base_modelo_poly = poly_transformer.fit_transform(base_modelo[cuantitativas_bm])

    # Combinar características originales con polinómicas
    base_modelo_combined = np.hstack([base_modelo, base_modelo_poly])

    # Estandarización de los datos
    scaler = StandardScaler()
    base_modelo_scaled = scaler.fit_transform(base_modelo_combined)
    base_modelo_final = pd.DataFrame(base_modelo_scaled, columns=[f'feature_{i}' for i in range(base_modelo_scaled.shape[1])])

    return base_modelo_final

# Interfaz de usuario en Streamlit
st.title('Predicción del Estado del Estudiante')

row = [48,1,7,1,9500,1,3,140.0,1,19,19,9,9,140.0,0,0,0,1,0,0,27,0,0,7,0,0,0.0,0,0,8,0,0,0.0,0,12.4,0.5,1.79,"Dropout"]
row_values = row[1:-1]  # Excluir el id (primer valor) y el Target (último valor)

# Solicitar entrada del usuario
# Entradas del usuario
input_data = {
    'Marital status': st.number_input('Estado civil', value=row_values[0]),
    'Application mode': st.number_input('Modo de aplicación', value=row_values[1]),
    'Application order': st.number_input('Orden de aplicación', value=row_values[2]),
    'Course': st.number_input('Curso', value=row_values[3]),
    'Daytime/evening attendance': st.number_input('Asistencia diurna/nocturna', value=row_values[4]),
    'Previous qualification': st.number_input('Calificación previa', value=row_values[5]),
    'Previous qualification (grade)': st.number_input('Nota de calificación previa', value=row_values[6]),
    'Nacionality': st.number_input('Nacionalidad', value=row_values[7]),
    "Mother's qualification": st.number_input('Calificación de la madre', value=row_values[8]),
    "Father's qualification": st.number_input('Calificación del padre', value=row_values[9]),
    "Mother's occupation": st.number_input('Ocupación de la madre', value=row_values[10]),
    "Father's occupation": st.number_input('Ocupación del padre', value=row_values[11]),
    'Admission grade': st.number_input('Nota de admisión', value=row_values[12]),
    'Displaced': st.number_input('Desplazado', value=row_values[13]),
    'Educational special needs': st.number_input('Necesidades educativas especiales', value=row_values[14]),
    'Debtor': st.number_input('Deudor', value=row_values[15]),
    'Tuition fees up to date': st.number_input('Cuotas al día', value=row_values[16]),
    'Gender': st.number_input('Género', value=row_values[17]),
    'Scholarship holder': st.number_input('Becado', value=row_values[18]),
    'Age at enrollment': st.number_input('Edad al momento de inscripción', value=row_values[19]),
    'International': st.number_input('Internacional', value=row_values[20]),
    'Curricular units 1st sem (credited)': st.number_input('Unidades curriculares 1er semestre acreditadas', value=row_values[21]),
    'Curricular units 1st sem (enrolled)': st.number_input('Unidades curriculares 1er semestre inscritas', value=row_values[22]),
    'Curricular units 1st sem (evaluations)': st.number_input('Unidades curriculares 1er semestre evaluadas', value=row_values[23]),
    'Curricular units 1st sem (approved)': st.number_input('Unidades curriculares 1er semestre aprobadas', value=row_values[24]),
    'Curricular units 1st sem (grade)': st.number_input('Nota del 1er semestre', value=row_values[25]),
    'Curricular units 1st sem (without evaluations)': st.number_input('Unidades curriculares 1er semestre sin evaluaciones', value=row_values[26]),
    'Curricular units 2nd sem (credited)': st.number_input('Unidades curriculares 2do semestre acreditadas', value=row_values[27]),
    'Curricular units 2nd sem (enrolled)': st.number_input('Unidades curriculares 2do semestre inscritas', value=row_values[28]),
    'Curricular units 2nd sem (evaluations)': st.number_input('Unidades curriculares 2do semestre evaluadas', value=row_values[29]),
    'Curricular units 2nd sem (approved)': st.number_input('Unidades curriculares 2do semestre aprobadas', value=row_values[30]),
    'Curricular units 2nd sem (grade)': st.number_input('Nota del 2do semestre', value=row_values[31]),
    'Curricular units 2nd sem (without evaluations)': st.number_input('Unidades curriculares 2do semestre sin evaluaciones', value=row_values[32]),
    'Unemployment rate': st.number_input('Tasa de desempleo', value=row_values[33]),
    'Inflation rate': st.number_input('Tasa de inflación', value=row_values[34]),
    'GDP': st.number_input('PIB', value=row_values[35])
}

if st.button("Predecir"):
    try:
        input_data = {key: float(value) for key, value in input_data.items()}
        input_df = pd.DataFrame([input_data])
        
        
        categoricas_bm = ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor',
                            'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Marital status',
                            'Application mode', 'Application order', 'Course', 'Previous qualification', 'Nacionality',
                            "Mother's qualification", "Father's qualification", "Mother's occupation",
                            "Father's occupation", ]
        
        input_df[categoricas_bm] = input_df[categoricas_bm].astype('category')
        cuantitativas_bm = [col for col in input_df.columns if col not in categoricas_bm]
        
        le = LabelEncoder()
        for col in categoricas_bm:
            input_df[col] = le.fit_transform(input_df[col].astype(str))
        data_prepared = aplicar_ingenieria_de_variables(input_df, cuantitativas_bm, categoricas_bm)
        yhat = modelo.predict(input_df)

        prediction_label = label_mapping.get(yhat[0], "Unknown")
        st.markdown(f"<p class='big-font'>Predicción: {prediction_label}</p>", unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Error en la entrada de datos: {str(e)}")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.rerun()