#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_auth
import base64
from dash import dcc, html
from IPython.display import Image
from dash.dependencies import Input, Output
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dash.exceptions import PreventUpdate
from flask import Flask, render_template
from dash import html
from pyproj import CRS


# In[83]:


emergency_df = pd.read_csv('C:/analysis/emergency.csv', encoding='cp949')
trace_30min_below = go.Bar(
    x=emergency_df['시도'],
    y=emergency_df['30분 미만'],
    name='30분 미만',
    text=emergency_df['30분 미만'])
trace_30min_to_2hr_below = go.Bar(
    x=emergency_df['시도'],
    y=emergency_df['30분~2시간 미만'],
    name='30분~2시간 미만',
    text=emergency_df['30분~2시간 미만'])
trace_2hr_to_4hr_below = go.Bar(
    x=emergency_df['시도'],
    y=emergency_df['2시간~4시간 미만'],
    name='2시간~4시간 미만',
    text=emergency_df['2시간~4시간 미만'])
trace_4hr_to_6hr_below = go.Bar(
    x=emergency_df['시도'],
    y=emergency_df['4시간~6시간 미만'],
    name='4시간~6시간 미만',
    text=emergency_df['4시간~6시간 미만'])
data = [trace_30min_below, trace_30min_to_2hr_below, trace_2hr_to_4hr_below, trace_4hr_to_6hr_below]
layout = go.Layout(
    title='<발병 후 의료시설 도착시간>',
    xaxis=dict(title='시도'),
    yaxis=dict(title='응급실 도착까지 소요시간'),
    plot_bgcolor='#E5ECF6',  # 여기서 수정
    paper_bgcolor='#E5ECF6',

    barmode='group')
fig = go.Figure(data=data, layout=layout)


# In[84]:


# 시도별 의사 직급별 합
df_doctors = pd.read_csv('data_doctor.csv', encoding='cp949')
third = px.bar(df_doctors, x='시도코드명', y=['의과일반의 인원수', '의과인턴 인원수', '의과레지던트 인원수', '의과전문의 인원수'],
                 title='<시도별 의사 직급에 따른 수>',
                 labels={'시도코드명': '시도', 'value': '의사 수', 'variable': '의사 직급'},
                 color_discrete_map={'의과일반': 'blue', '의과인턴': 'orange', '의과레지던트': 'green', '의과전문의': 'red'},
                 barmode='stack'
                )
third.update_layout(
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6'
)


# In[85]:


# 지역별 상급종합병원 비율
filtered_df = df_doctors[df_doctors['종별코드명'] == '상급종합']
hospital_count_by_region = filtered_df.groupby('시도코드명')['요양기관명'].count().reset_index(name='병원 수')
threshold = 0.05
small_values = hospital_count_by_region[hospital_count_by_region['병원 수'] / hospital_count_by_region['병원 수'].sum() < threshold]
hospital_count_by_region.loc[hospital_count_by_region['시도코드명'].isin(small_values['시도코드명']), '시도코드명'] = '기타(울산, 강원, 대전, 광주, 세종, 제주)'
hospital_count_by_region = hospital_count_by_region.groupby('시도코드명')['병원 수'].sum().reset_index()
sorted_df = hospital_count_by_region[hospital_count_by_region['시도코드명'] != '기타'].sort_values(by='병원 수', ascending=True)
first = px.pie(sorted_df, values='병원 수', names='시도코드명', title='<지역별 상급종합병원 비율>')
first.update_layout(
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6'
)


# In[86]:


# 지역별 종합병원 비율
filtered_df2 = df_doctors[df_doctors['종별코드명'] == '종합병원']
hospital_count_by_region2 = filtered_df2.groupby('시도코드명')['요양기관명'].count().reset_index(name='병원 수')
threshold = 0.05
small_values = hospital_count_by_region2[hospital_count_by_region2['병원 수'] / hospital_count_by_region2['병원 수'].sum() < threshold]
hospital_count_by_region2.loc[hospital_count_by_region2['시도코드명'].isin(small_values['시도코드명']), '시도코드명'] = '기타(강원, 제주, 울산, 세종, 대전, 대구)'
hospital_count_by_region2 = hospital_count_by_region2.groupby('시도코드명')['병원 수'].sum().reset_index()
sorted_df2 = hospital_count_by_region2[hospital_count_by_region2['시도코드명'] != '기타'].sort_values(by='병원 수', ascending=True)
second = px.pie(sorted_df2, values='병원 수', names='시도코드명', title='<지역별 종합병원 비율>')
second.update_layout(
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6'
)


# In[87]:


# 인구수 대비 병원수
df_hospitals = pd.read_csv('data_road.csv', encoding='cp949')
df_population = pd.read_csv('data_pop.csv', encoding='cp949')
df_hospitals = df_hospitals[['시도코드명', '요양기관명', '종별코드명', 'Longitude', 'Latitude']]
df_hospitals['시도코드명'] = df_hospitals['시도코드명'].replace({'전북': '전라', '전남': '전라'})
df_hospitals['시도코드명'] = df_hospitals['시도코드명'].replace({'충남': '충청', '충북': '충청'})
df_hospitals['시도코드명'] = df_hospitals['시도코드명'].replace({'경남': '경상', '경북': '경상'})
unique_locations = df_hospitals.drop_duplicates(subset=['Longitude', 'Latitude'])
gdf_hospitals = gpd.GeoDataFrame(unique_locations, geometry=gpd.points_from_xy(unique_locations['Longitude'], unique_locations['Latitude']))
gdf_hospitals = gdf_hospitals.dropna(subset=['시도코드명'])
hospitals_by_region = gdf_hospitals.groupby('시도코드명').size().reset_index(name='병원 수')
merged_df = pd.merge(hospitals_by_region, df_population, how='left', on='시도코드명')
merged_df = pd.merge(merged_df, gdf_hospitals[['시도코드명', 'Longitude', 'Latitude']], how='left', on='시도코드명')
merged_df['인구 대비 병원 수'] = merged_df['인구수'] / merged_df['병원 수']
merged_df['geometry'] = gpd.points_from_xy(merged_df['Longitude'], merged_df['Latitude'])
HEXAGON_LAYER_DATA = merged_df[['Longitude', 'Latitude', '인구 대비 병원 수']]
# HexagonLayer 생성
layer = pdk.Layer(
    "HeatmapLayer",
    HEXAGON_LAYER_DATA,
    get_position=["Longitude", "Latitude"],
    auto_highlight=True,
    elevation_scale=1000,  # 필요에 따라 조절
    pickable=False,
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6',
)
view_state = pdk.ViewState(latitude=36, longitude=127, zoom=5)
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
)


# In[88]:


emergency_person_df1 = pd.read_csv('C:/analysis/emergency_medical_person.csv', encoding='cp949')
fifth1 = go.Figure()
fifth1.add_trace(go.Bar(
    y=emergency_person_df1['지역'],
    x=emergency_person_df1['응급실전담 전문의 수'],
    name='응급실 전담 전문의',
    orientation='h'))
fifth1.add_trace(go.Bar(
    y=emergency_person_df1['지역'],
    x=emergency_person_df1['응급실전담 간호사 수'],
    name='응급실 전담 간호사',
    orientation='h'))
fifth1.update_layout(
    barmode='stack',
    title='<시도별 응급의료기관 근무인력 현황>',
    xaxis=dict(title='근무인력 수'),
    yaxis=dict(title='지역'),
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6',)


# In[89]:


emergency_person_df1 = pd.read_csv('C:/analysis/dash_emergency.csv', encoding='cp949')
fifth2 = go.Figure()
fifth2.add_trace(go.Bar(
    y=emergency_person_df1['지역'],
    x=emergency_person_df1['응급실전담 전문의 수'],
    name='응급실 전담 전문의',
    orientation='h'))
fifth2.add_trace(go.Bar(
    y=emergency_person_df1['지역'],
    x=emergency_person_df1['응급실전담 간호사 수'],
    name='응급실 전담 간호사',
    orientation='h'))
fifth2.update_layout(
    barmode='stack',
    title='<시도별 인구 천명당 응급의료기관 근무인력 현황>',
    xaxis=dict(title='근무인력 수'),
    yaxis=dict(title='지역'),
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6',)


# In[90]:


emergency_person_df2= pd.read_csv('C:/analysis/emergency_medical_person.csv', encoding='cp949')
sixth = go.Figure()
sixth.add_trace(go.Bar(
    y=emergency_person_df2['지역'],
    x=emergency_person_df2['응급의료기관 1개소 당 응급의학전문의 수'],
    name='응급실 전담 전문의',
    orientation='h'))
sixth.add_trace(go.Bar(
    y=emergency_person_df2['지역'],
    x=emergency_person_df2['응급의료기관 1개소 당 간호사 수'],
    name='응급실 전담 간호사',
    orientation='h'))
sixth.update_layout(
    barmode='stack',
    title='<응급의료기관 1개소 당 시도별 응급의료기관 근무인력 현황>',
    xaxis=dict(title='근무인력 수'),
    yaxis=dict(title='지역'),
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6',)


# In[91]:


df = pd.read_excel('emergency_count_2022.xlsx', header = 1)
geojson_path = "sido.json"  # 실제 파일 이름으로 업데이트하세요.
gdf = gpd.read_file(geojson_path)
merged_data = gdf.merge(df, how='left', left_on='CTP_KOR_NM', right_on='분류')

fig5 = px.choropleth(merged_data,
                    geojson=merged_data.geometry,
                    locations=merged_data.index,
                    color='계',
                    hover_name='분류',
                    title='<전국 응급의료시설 현황>',
                    color_continuous_scale='YlOrRd')
fig5.update_layout(
    plot_bgcolor='#E5ECF6',
    paper_bgcolor='#E5ECF6'
)
view_state = pdk.ViewState(latitude=36, longitude=127, zoom=20)


# In[92]:


# 대시 앱 생성
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#E5ECF6'}, children=[
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='1', value='tab-1'),
        dcc.Tab(label='2', value='tab-2'),
        dcc.Tab(label='3', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])


# In[93]:


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def update_tab(selected_tab):
    if selected_tab == 'tab-1':
        return html.Div([
            html.H1("의료시설 수도권 집중현상", style={'fontSize': '50px', 'color':'#387482', 'text-align': 'center'}),
            # 그래프 1 (왼쪽)
            html.Div([
                html.H1(style={'fontSize': '10px', 'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=first, config={'displayModeBar': False}),
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            # 그래프 2 (오른쪽)
            html.Div([
                html.H1(style={'fontSize': '10px', 'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=second, config={'displayModeBar': False}),
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                html.H1(style={'fontSize': '10px', 'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=third, config={'displayModeBar': False}),
            ]),
            html.Div([
                html.H1("<인구 대비 병원 수 지도>", style={'fontSize': '20px', 'font-family': 'Arial, sans-serif', 'text-align': 'center'}),
                html.Iframe(srcDoc=open('hexagon_map.html').read(), width='100%', height='600px'),
                html.Hr(),
            ]),
        ])
    elif selected_tab == 'tab-2':
        return html.Div([
            html.H2("의료시설 수도권 집중현상", style={'fontSize': '50px', 'color':'#387482', 'text-align': 'center'}),
            html.Div([
                dcc.Graph(figure=fig5, config={'displayModeBar': False}),
            ]),
            html.Hr(),
        ]),
    elif selected_tab == 'tab-3':
        return html.Div([
            html.H3("의료시설 수도권 집중현상", style={'fontSize': '50px', 'color':'#387482', 'text-align': 'center'}),
            html.Div([
                html.H3(style={'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=fig, config={'displayModeBar': False}),
            ]),
            html.Div([
                html.H3(style={'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=fifth1, config={'displayModeBar': False}),
            ]),
            html.Div([
                html.H3(style={'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=fifth2, config={'displayModeBar': False}),
            ]),
            html.Div([
                html.H3(style={'font-family': 'Arial, sans-serif'}),
                dcc.Graph(figure=sixth, config={'displayModeBar': False}),
            ]),
        ])


if __name__ == '__main__':
    app.run_server(debug=True, port=9030)


# In[ ]:





# In[ ]:





# In[ ]:




