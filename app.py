import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model import load_model_components, estimate_y

# Load pre-trained model and components
model, poly_features, q, mapping, reverse_mapping = load_model_components()

# Load data (assuming you still need this for plotting)
data = pd.read_csv('data.csv')

# Streamlit app
st.title('Estimator OB Price Based on HDVD')

# Input widgets
cont_nominal = st.selectbox('CONT:', options=['PAMA', 'SIS', 'BUMA', 'PPA', 'RA'])
hd = st.number_input('HD:')
vd = st.number_input('VD:')

# Prediction button
if st.button('Calculate and Plot'):
    cont = mapping[cont_nominal]
    y_pred, lower, upper = estimate_y(model, poly_features, q, vd, hd, cont)

    st.write(f'**Estimated price:** {round(y_pred)}')
    st.write(f'**95% prediction interval:** [{round(lower)}, {round(upper)}]')

    # Plot the results
    fig = go.Figure()

    # Add actual data points
    actual_data = data[data['CONT'] == cont_nominal]
    fig.add_trace(go.Scatter3d(
        x=actual_data['VD'],
        y=actual_data['HD'],
        z=actual_data['Value_'],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.8),
        name='Actual'
    ))

    # Add predicted point
    fig.add_trace(go.Scatter3d(
        x=[vd],
        y=[hd],
        z=[y_pred],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        name='Predicted'
    ))

    fig.update_layout(
        scene=dict(xaxis_title='VD', yaxis_title='HD', zaxis_title='Value'),
        width=800,
        height=600,
        title=f'Actual vs Predicted Values for {cont_nominal}'
    )

    st.plotly_chart(fig)