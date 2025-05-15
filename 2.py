import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import plotly.graph_objects as go

np.random.seed(42)
df = pd.DataFrame(np.random.normal(50, 7, 500), columns=['underlying_ability'])
df['test'] = df['underlying_ability'] + np.random.normal(0, 10, 500)
df['self_estimate'] = df['underlying_ability'] + np.random.normal(0, 15, 500)

df['test'] = df['test'].clip(0, 100)
df['self_estimate'] = df['self_estimate'].clip(0, 100)

model = ols("self_estimate ~ test", df).fit()
df['regression'] = model.predict(df['test'])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['test'], 
    y=df['self_estimate'],
    mode='markers',
    marker=dict(opacity=0.2),
    name='Individual estimates'
))

fig.add_trace(go.Scatter(
    x=df['test'],
    y=df['regression'],
    mode='lines',
    line=dict(color='blue', width=2),
    name='Regression line'
))

fig.update_layout(
    title='Dunning-Kruger Effect Simulation',
    xaxis_title='Test Score (Actual Performance)',
    yaxis_title='Self Estimate',
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True
)

fig.show()