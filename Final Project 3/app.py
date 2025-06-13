import os
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import ee
import geemap.foliumap as geemap
import geopandas as gpd
import time
import threading
from queue import Queue

# Initialize Earth Engine
try:
    ee.Initialize(project='enviro-454206')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='enviro-454206')

# API Keys and Constants
MISTRAL_API_KEY = "bAVPgWJVSNP5pvmJZeSehN8OD9KapHHv"  # Replace with your actual key
GEMINI_API_KEY = "AIzaSyA5l-6R3H1lBUDRLFeyCZctS1hIS0YAc3Y"  # Replace with your actual key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
CORPUS_FOLDER = "CORPUS"  # Ensure this folder exists with state files

state_corpus_files = {
    "Andhra Pradesh": "Andhra Pradesh_training_corpus.txt",
    "Arunachal Pradesh": "Arunachal Pradesh_training_corpus.txt",
    "Assam": "Assam_training_corpus.txt",
    "Bihar": "Bihar_training_corpus.txt",
    "Chhattisgarh": "Chhattisgarh_training_corpus.txt",
    "Goa": "Goa_training_corpus.txt",
    "Gujarat": "Gujarat_training_corpus.txt",
    "Haryana": "Haryana_training_corpus.txt",
    "Himachal Pradesh": "Himachal Pradesh_training_corpus.txt",
    "Jharkhand": "Jharkhand_training_corpus.txt",
    "Karnataka": "Karnataka_training_corpus.txt",
    "Kerala": "Kerala_training_corpus.txt",
    "Madhya Pradesh": "Madhya Pradesh_training_corpus.txt",
    "Maharashtra": "Maharashtra_training_corpus.txt",
    "Manipur": "Manipur_training_corpus.txt",
    "Meghalaya": "Meghalaya_training_corpus.txt",
    "Mizoram": "Mizoram_training_corpus.txt",
    "Nagaland": "Nagaland_training_corpus.txt",
    "Odisha": "Odisha_training_corpus.txt",
    "Punjab": "Punjab_training_corpus.txt",
    "Rajasthan": "Rajasthan_training_corpus.txt",
    "Sikkim": "Sikkim_training_corpus.txt",
    "Tamil Nadu": "Tamil Nadu_training_corpus.txt",
    "Telangana": "Telangana_training_corpus.txt",
    "Tripura": "Tripura_training_corpus.txt",
    "Uttar Pradesh": "Uttar Pradesh_training_corpus.txt",
    "Uttarakhand": "Uttarakhand_training_corpus.txt",
    "West Bengal": "West Bengal_training_corpus.txt"
}

# Dynamic World land cover classes
DYNAMIC_WORLD_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice"
]

# Helper Functions
def extract_year(query):
    year_dict = {}
    states = extract_states_from_query(query)
    if not states:
        states = ["default"]
    
    for state in states:
        year_dict[state] = []

    last_n_years_match = re.search(r"last\s+(\d+)\s+years?", query, re.IGNORECASE)
    if last_n_years_match:
        n_years = int(last_n_years_match.group(1))
        current_year = 2024
        start_year = max(2015, current_year - n_years + 1)
        for state in states:
            year_dict[state] = [str(y) for y in range(start_year, current_year + 1)]
        return year_dict

    range_match = re.search(r"(\d{4})\s*(?:to|-)\s*(\d{4})", query, re.IGNORECASE)
    if range_match:
        start_year, end_year = map(int, range_match.groups())
        start_year = max(2015, start_year)
        end_year = min(2024, end_year)
        if start_year <= end_year:
            for state in states:
                year_dict[state] = [str(y) for y in range(start_year, end_year + 1)]
    
    multiple_years = re.findall(r"\b(\d{4})\b", query)
    if multiple_years and not range_match:
        valid_years = [y for y in multiple_years if 2015 <= int(y) <= 2024]
        if valid_years:
            for state in states:
                year_dict[state] = sorted(set(valid_years))
    
    for state in states:
        state_pattern = rf"\b{re.escape(state)}\s*(\d{{4}})?\b"
        match = re.search(state_pattern, query, re.IGNORECASE)
        if match and match.group(1) and not year_dict[state]:
            year_dict[state] = [match.group(1)] if 2015 <= int(match.group(1)) <= 2024 else []
    
    for state in states:
        if not year_dict[state]:
            year_dict[state] = ["2024"]
    
    return year_dict

def extract_states_from_query(query):
    return [state for state in state_corpus_files.keys() if re.search(rf"\b{state}\b", query, re.IGNORECASE)]

def extract_metrics_from_query(query):
    query_lower = query.lower()
    metrics = []

    if "land cover" in query_lower:
        return DYNAMIC_WORLD_CLASSES

    if any(metric.lower() in query_lower for metric in ["ndvi", "nbr", "evi"]):
        return ["NDVI", "NBR", "EVI"]

    if any(metric.lower() in query_lower for metric in ["ndmi", "mndwi"]):
        return ["NDMI", "MNDWI", "NDVI", "NBR", "EVI"]

    if "trees" in query_lower:
        return ["NDVI", "NBR", "EVI"] + DYNAMIC_WORLD_CLASSES

    if "water" in query_lower:
        return DYNAMIC_WORLD_CLASSES + ["MNDWI"]

    return ["NDVI"]

def clean_response(response):
    response = re.sub(r"DynamicWorld\s+([a-z_]+)\s*:\s*([\d.]+)", r"\1: \2", response)
    patterns = [
        r"Sentinel2\s+([A-Za-z0-9]+)\s*(?:at|is|:)?\s*([\d.]+)"
    ]
    for pattern in patterns:
        response = re.sub(pattern, r"\1: \2", response)
    return response.strip()

def call_mistral_saba(api_url, api_key, corpus, query, states, metrics=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    instruction = (
        "Provide a response using only numerical values from the corpus. "
        "List metrics and values (e.g., '2023 NDVI: 0.415 Kerala' or '2023 water: 0.2 Kerala') with corresponding years and states. "
        "For land cover, include requested Dynamic World classes (water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice) with values summing to 1.0 per year and state, prefixed with 'DynamicWorld' (e.g., 'DynamicWorld water: 0.1'). "
        "If metrics are specified, only include those; otherwise, include all requested data. "
        "If multiple states or years are requested, provide data for each state-year combination separately. "
        "If no data, return 'No relevant data'. "
        "End with: 'Data sourced from Sentinel-2 and Dynamic World.'"
    )
    year_dict = extract_year(query)
    year_str = ""
    for state in states:
        years = year_dict.get(state, ["2024"])
        year_str += f"{state}: {', '.join(years)}; "
    
    metric_instruction = f"Metrics requested: {', '.join(metrics)}" if metrics else "All available metrics"
    payload = {
        "model": "mistral-saba-2502",
        "messages": [
            {"role": "system", "content": f"You are an AI trained on environmental data. {instruction}\n{metric_instruction}"},
            {"role": "user", "content": f"Context: {corpus}\nQuery: {query} for {', '.join(states)}"}
        ]
    }
    try:
        for attempt in range(3):
            try:
                response = requests.post(api_url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    return f"API Error: {str(e)}"
                time.sleep(1)
        raw_response = response.json()
        print(f"Raw API Response: {raw_response}")
        return raw_response.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

def generate_visualization(mistral_values, states, year_dict, query, requested_metrics):
    figures = []
    print(f"Initial Mistral Values: {mistral_values}")
    print(f"States: {states}")
    print(f"Year Dict: {year_dict}")
    print(f"Query: {query}")
    print(f"Requested Metrics: {requested_metrics}")

    pattern = r"(\d{4})\s*(?:DynamicWorld\s+)?(\w+)\s*:\s*([\d.]+)\s*(\w+)"
    matches = re.findall(pattern, mistral_values)
    print(f"Parsed Matches: {matches}")

    if not matches:
        print("No valid data found in Mistral response for visualization.")
        return figures

    # Populate data_by_state_year
    data_by_state_year = {}
    checkout_corpus_data = ""
    for state in states:
        corpus_file_path = os.path.join(CORPUS_FOLDER, state_corpus_files[state])
        if os.path.exists(corpus_file_path):
            with open(corpus_file_path, "r", encoding="utf-8") as f:
                checkout_corpus_data += f"\n--- {state} ---\n" + f.read()

    for state in states:
        years = year_dict.get(state, ["2024"])
        if not isinstance(years, list):
            years = [years]
        years = sorted(set(str(y) for y in years if 2015 <= int(y) <= 2024))
        data_by_state_year[state] = {}
        
        for year in years:
            values = [0.0] * len(requested_metrics)
            found_data = False

            for year_match, metric, value, state_match in matches:
                if state_match == state and year_match == year and metric in requested_metrics:
                    idx = requested_metrics.index(metric)
                    values[idx] = float(value)
                    found_data = True
                    data_by_state_year[state][year] = values

            if not found_data:
                specific_query = f"Environmental data for {', '.join(requested_metrics)} in {state} {year}"
                mistral_response = call_mistral_saba(MISTRAL_API_URL, MISTRAL_API_KEY,
                                                   checkout_corpus_data, specific_query,
                                                   [state], requested_metrics)
                print(f"Mistral Response for {state} {year}: {mistral_response}")
                values = [0.0] * len(requested_metrics)
                for idx, metric in enumerate(requested_metrics):
                    metric_pattern = rf"(?:DynamicWorld\s+)?{metric}\s*:\s*([\d.]+)\s*(?:\n|\s*{state})?"
                    match = re.search(metric_pattern, mistral_response)
                    if match:
                        values[idx] = float(match.group(1))
                        found_data = True
                if found_data:
                    data_by_state_year[state][year] = values
                print(f"Data for {state} {year}: {values}")

    print(f"Data by State Year: {data_by_state_year}")

    # 1. Bar chart for comparison queries (NDVI/NBR across multiple states)
    if len(states) > 1 and "compare" in query.lower() and "land cover" not in query.lower():
        metrics = [(year, metric, float(value) if value else 0.0, state)
                  for year, metric, value, state in matches
                  if metric in ["NDVI", "NBR"] and state in states]
        print(f"Bar Chart Matches: {metrics}")
        if metrics:
            df = pd.DataFrame(metrics, columns=["Year", "Metric", "Value", "State"])
            print(f"Bar Chart DataFrame: {df}")
            if not df.empty:
                pivot_df = df.pivot_table(index=["State", "Year"], columns="Metric",
                                         values="Value", aggfunc="first").reset_index().fillna(0.0)
                pivot_df['State_Year'] = pivot_df['State'] + ' (' + pivot_df['Year'] + ')'
                print(f"Pivot DataFrame: {pivot_df}")

                fig = go.Figure()
                for metric in ["NDVI", "NBR"]:
                    if metric in pivot_df.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_df['State_Year'],
                            y=pivot_df[metric],
                            name=metric,
                            marker_color='rgb(55, 83, 109)' if metric == "NDVI" else 'rgb(26, 118, 255)',
                            opacity=0.9,
                            text=[f'{v:.2f}' for v in pivot_df[metric]],
                            textposition='auto'
                        ))

                fig.update_layout(
                    title={'text': 'Environmental Metrics Comparison', 'x': 0.5, 'xanchor': 'center'},
                    xaxis_title='States and Years',
                    yaxis_title='Values',
                    barmode='group',
                    xaxis_tickangle=45,
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    margin=dict(l=50, r=50, t=80, b=100),
                    yaxis=dict(gridcolor='rgba(200,200,200,0.3)')
                )
                max_value = max(pivot_df.get("NDVI", [0]).max(), pivot_df.get("NBR", [0]).max(), 0.1)
                fig.update_yaxes(range=[0, max_value * 1.2])
                figures.append(fig)
            else:
                print("No valid data for bar chart after filtering")

    # 2. Bar charts for land cover or specific metrics
    for state in data_by_state_year:
        years = year_dict.get(state, ["2024"])
        if not isinstance(years, list):
            years = [years]
        years = sorted(set(str(y) for y in years if 2015 <= int(y) <= 2024))

        for year in years:
            if year in data_by_state_year[state]:
                values = data_by_state_year[state][year]
                if any(v > 0 for v in values):
                    title = f'Metrics for {state} ({year})'
                    fig = go.Figure(data=[
                        go.Bar(
                            x=requested_metrics,
                            y=values,
                            text=[f'{v:.2f}' for v in values],
                            textposition='auto',
                            marker_color=px.colors.qualitative.Plotly,
                            opacity=0.9
                        )
                    ])
                    fig.update_layout(
                        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
                        xaxis_title='Metrics',
                        yaxis_title='Proportion',
                        xaxis_tickangle=45,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        margin=dict(l=50, r=50, t=80, b=100),
                        yaxis=dict(gridcolor='rgba(200,200,200,0.3)')
                    )
                    max_value = max(values + [0.1])
                    fig.update_yaxes(range=[0, max_value * 1.5])
                    figures.append(fig)

        # Combined bar chart for multiple years
        if len(years) > 1:
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, year in enumerate(years):
                if year in data_by_state_year[state]:
                    values = data_by_state_year[state][year]
                    fig.add_trace(go.Bar(
                        x=requested_metrics,
                        y=values,
                        name=f"{year}",
                        text=[f'{v:.2f}' for v in values],
                        textposition='auto',
                        marker_color=colors[i % len(colors)],
                        opacity=0.9
                    ))

            fig.update_layout(
                title={'text': f'Metrics Comparison for {state} ({min(years)}–{max(years)})', 'x': 0.5, 'xanchor': 'center'},
                xaxis_title='Metrics',
                yaxis_title='Proportion',
                barmode='group',
                xaxis_tickangle=45,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=100),
                yaxis=dict(gridcolor='rgba(200,200,200,0.3)')
            )
            max_value = max(max(data_by_state_year[state][y]) for y in years if y in data_by_state_year[state]) + 0.1
            fig.update_yaxes(range=[0, max_value * 1.2])
            figures.append(fig)

    # 3. Pie charts for land cover
    if "land cover" in query.lower():
        generated_charts = set()
        for state in states:
            years = year_dict.get(state, ["2024"])
            if not isinstance(years, list):
                years = [years]
            years = sorted(set(str(y) for y in years if 2015 <= int(y) <= 2024))
            
            if state not in data_by_state_year:
                print(f"No land cover data for {state}")
                continue
                
            for year in years:
                chart_key = f"{state}_{year}"
                if chart_key in generated_charts:
                    print(f"Skipping duplicate pie chart for {state} {year}")
                    continue
                    
                if year in data_by_state_year[state]:
                    values = data_by_state_year[state][year]
                    if not any(v > 0 for v in values):
                        print(f"No non-zero land cover values for {state} {year}")
                        continue
                        
                    total = sum(values)
                    if total > 0 and abs(total - 1.0) > 0.01:
                        values = [v / total for v in values]
                        
                    non_zero_metrics = [m for m, v in zip(requested_metrics, values) if v > 0]
                    non_zero_values = [v for v in values if v > 0]
                    
                    if non_zero_metrics:
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=non_zero_metrics,
                                values=non_zero_values,
                                textinfo='label+percent',
                                insidetextorientation='radial',
                                marker=dict(colors=px.colors.qualitative.Plotly),
                                hole=0.3
                            )
                        ])
                        fig.update_layout(
                            title={'text': f'Land Cover Distribution for {state} ({year})', 'x': 0.5, 'xanchor': 'center'},
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=12),
                            margin=dict(l=50, r=50, t=80, b=50)
                        )
                        figures.append(fig)
                        generated_charts.add(chart_key)
                        print(f"Generated Pie Chart for {state} {year}: {non_zero_metrics}")
                    else:
                        print(f"No non-zero land cover metrics for {state} {year}")

    # 4. Line graph for multiple years (single state)
    if len(states) == 1:
        state = states[0]
        years = year_dict.get(state, ["2024"])
        if isinstance(years, list) and len(years) > 1:
            years = [str(y) for y in years if 2015 <= int(y) <= 2024]
            if years and state in data_by_state_year:
                fig = go.Figure()
                plotted = False
                colors = px.colors.qualitative.Plotly
                for i, metric in enumerate(requested_metrics):
                    values = [data_by_state_year[state].get(y, [0.0] * len(requested_metrics))[requested_metrics.index(metric)]
                             for y in years]
                    if any(v > 0 for v in values):
                        fig.add_trace(go.Scatter(
                            x=[int(y) for y in years],
                            y=values,
                            mode='lines+markers',
                            name=metric,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8),
                            text=[f'{v:.2f}' for v in values],
                            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
                        ))
                        plotted = True
                if plotted:
                    fig.update_layout(
                        title={'text': f'Metrics Trends for {state} ({min(years)}–{max(years)})', 'x': 0.5, 'xanchor': 'center'},
                        xaxis_title='Year',
                        yaxis_title='Proportion',
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        margin=dict(l=50, r=50, t=80, b=50),
                        xaxis=dict(tickvals=[int(y) for y in years]),
                        yaxis=dict(gridcolor='rgba(200,200,200,0.3)', range=[0, 1.0])
                    )
                    figures.append(fig)
                else:
                    print(f"No data for {state} across years {years}")

    # 5. Combined line graph for multiple states
    if len(states) > 1:
        years = set()
        for state in states:
            years.update([str(y) for y in year_dict.get(state, ["2024"]) if 2015 <= int(y) <= 2024])
        if years:
            for metric in requested_metrics:
                fig = go.Figure()
                plotted = False
                colors = px.colors.qualitative.Plotly
                for i, state in enumerate(states):
                    state_years = [str(y) for y in year_dict.get(state, ["2024"]) if y in years]
                    values = [data_by_state_year.get(state, {}).get(y, [0.0] * len(requested_metrics))[requested_metrics.index(metric)]
                             for y in state_years]
                    if any(v > 0 for v in values):
                        fig.add_trace(go.Scatter(
                            x=[int(y) for y in state_years],
                            y=values,
                            mode='lines+markers',
                            name=f"{state} ({metric})",
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8),
                            text=[f'{v:.2f}' for v in values],
                            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
                        ))
                        plotted = True
                if plotted:
                    year_range = f"{min(years)}–{max(years)}" if len(years) > 1 else years.pop()
                    fig.update_layout(
                        title={'text': f'{metric} Comparison for {", ".join(states)} ({year_range})', 'x': 0.5, 'xanchor': 'center'},
                        xaxis_title='Year',
                        yaxis_title='Proportion',
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        margin=dict(l=50, r=50, t=80, b=50),
                        xaxis=dict(tickvals=[int(y) for y in years]),
                        yaxis=dict(gridcolor='rgba(200,200,200,0.3)', range=[0, 1.0])
                    )
                    figures.append(fig)
                else:
                    print(f"No data for {metric} across states {states}")

    print(f"Generated Figures: {len(figures)}")
    return figures

def call_gemini(api_key, context, query, states, mistral_values):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    instruction = (
        "Provide a detailed response using only the specific numerical values provided in the context (mistral_values). "
        "List metric names and their exact values (e.g., 'NDVI: 0.415', 'trees: 0.654') from the dataset. Don't mention Sentinel or Dynamic World anywhere."
        "For each metric, include a paragraph (at least 50 words) explaining its significance, what the value indicates about the state's environment, and how it compares to typical ranges or other states if multiple are provided. "
        "Do not provide theoretical answers or values not present in the mistral_values. "
        "If multiple states are mentioned, structure the response with clear headings for each state and compare their metrics. "
        "If no relevant data is found, respond only with 'No relevant data'. "
        "End the response with: 'Data sourced from Sentinel-2 and Dynamic World.'"
    )
    context_with_values = f"Context: {context}\nMistral Values: {mistral_values}\nQuery: {query} for {', '.join(states)}\n{instruction}"
    response = model.generate_content(context_with_values)
    return clean_response(response.text) if hasattr(response, "text") else "Error generating response"

def generate_report(query, detected_states, year_dict, checkout_corpus_data, mistral_values):
    report = call_gemini(GEMINI_API_KEY, checkout_corpus_data, query, detected_states, mistral_values)
    return report

def generate_map(states, year_dict, query, result_queue):
    try:
        shapefile_path = "/Users/gopikrishna/Desktop/Final Project 3/SHAPE/gadm41_IND_1.shp"
        if not os.path.exists(shapefile_path):
            result_queue.put((None, f"Shapefile not found at {shapefile_path}"))
            return

        gdf = gpd.read_file(shapefile_path)
        gdf["NAME_1"] = gdf["NAME_1"].str.title()
        state_geoms = {}
        for state in states:
            state_gdf = gdf[gdf["NAME_1"] == state]
            if state_gdf.empty:
                print(f"No geometry found for {state}")
                continue
            state_geoms[state] = ee.Geometry(state_gdf.geometry.iloc[0].__geo_interface__)

        m = geemap.Map(zoom=7, height=400)
        requested_metrics = extract_metrics_from_query(query)
        for state, geom in state_geoms.items():
            boundary = ee.Feature(geom, {"style": {"color": "black", "width": 2}})
            m.addLayer(boundary, {"style": "outline"}, f"{state} Boundary")

            year = year_dict[state][0] if isinstance(year_dict[state], list) else year_dict[state]
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            s2_collection = None
            for attempt in range(3):
                try:
                    s2_collection = (
                        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                        .filterBounds(geom)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                        .sort("system:time_start", False)
                    )
                    size = s2_collection.size().getInfo()
                    if size > 0:
                        break
                    print(f"Attempt {attempt + 1}: No Sentinel-2 data for {state} {year}")
                    time.sleep(1)
                except Exception as e:
                    if attempt == 2:
                        print(f"Failed to fetch Sentinel-2 for {state} {year}: {str(e)}")
                        s2_collection = None
                        break
                    time.sleep(1)

            if not s2_collection or s2_collection.size().getInfo() == 0:
                print(f"No valid Sentinel-2 data for {state} {year}")
                continue

            s2 = s2_collection.mosaic().clip(geom) if s2_collection.size().getInfo() > 1 else s2_collection.first().clip(geom)
            
            if "NDVI" in requested_metrics:
                ndvi = s2.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{state}")
                m.addLayer(ndvi, {"min": 0, "max": 1, "palette": ["red", "yellow", "green"]}, f"NDVI ({state}, {year})")
            
            if "NBR" in requested_metrics:
                nbr = s2.normalizedDifference(["B8", "B12"]).rename(f"NBR_{state}")
                m.addLayer(nbr, {"min": -1, "max": 1, "palette": ["blue", "white", "red"]}, f"NBR ({state}, {year})")
            
            if "EVI" in requested_metrics:
                nir = s2.select("B8")
                red = s2.select("B4")
                blue = s2.select("B2")
                evi = nir.subtract(red).multiply(2.5).divide(
                    nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
                ).rename(f"EVI_{state}")
                m.addLayer(evi, {"min": 0, "max": 1, "palette": ["red", "yellow", "green"]}, f"EVI ({state}, {year})")
            
            if "NDMI" in requested_metrics:
                nir = s2.select("B8")
                swir1 = s2.select("B11")
                ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename(f"NDMI_{state}")
                m.addLayer(ndmi, {"min": -1, "max": 1, "palette": ["brown", "white", "blue"]}, f"NDMI ({state}, {year})")
            
            if "MNDWI" in requested_metrics:
                green = s2.select("B3")
                swir1 = s2.select("B11")
                mndwi = green.subtract(swir1).divide(green.add(swir1)).rename(f"MNDWI_{state}")
                m.addLayer(mndwi, {"min": -1, "max": 1, "palette": ["brown", "white", "blue"]}, f"MNDWI ({state}, {year})")
            
            if any(metric in DYNAMIC_WORLD_CLASSES for metric in requested_metrics):
                dw_collection = None
                for attempt in range(3):
                    try:
                        dw_collection = (
                            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                            .filterBounds(geom)
                            .filterDate(start_date, end_date)
                            .sort("system:time_start", False)
                        )
                        size = dw_collection.size().getInfo()
                        if size > 0:
                            break
                        print(f"Attempt {attempt + 1}: No Dynamic World data for {state} {year}")
                        time.sleep(1)
                    except Exception as e:
                        if attempt == 2:
                            print(f"Failed to fetch Dynamic World for {state} {year}: {str(e)}")
                            dw_collection = None
                            break
                        time.sleep(1)

                if dw_collection and dw_collection.size().getInfo() > 0:
                    land_cover = dw_collection.mosaic().clip(geom).select("label").rename(f"Land_Cover_{state}")
                    land_cover_viz = {
                        "min": 0,
                        "max": 8,
                        "palette": ["419BDF", "397D49", "88B053", "7A87C6", "E49635", "DFC35A", "C4281B", "A59B8F", "B39FE1"]
                    }
                    m.addLayer(land_cover, land_cover_viz, f"Land Cover ({state}, {year})")
                else:
                    print(f"No valid Dynamic World data for {state} {year}")

        if state_geoms:
            m.centerObject(next(iter(state_geoms.values())), 7)
            result_queue.put((m, None))
        else:
            result_queue.put((None, "No valid state geometries found"))
    except Exception as e:
        result_queue.put((None, f"Map generation failed: {str(e)}"))

def generate_comparative_maps(states, year_dict, query, requested_metrics, result_queue):
    try:
        shapefile_path = "/Users/gopikrishna/Desktop/Final Project 3/SHAPE/gadm41_IND_1.shp"
        if not os.path.exists(shapefile_path):
            result_queue.put((None, f"Shapefile not found at {shapefile_path}"))
            return

        gdf = gpd.read_file(shapefile_path)
        gdf["NAME_1"] = gdf["NAME_1"].str.title()
        state_geoms = {}
        for state in states:
            state_gdf = gdf[gdf["NAME_1"] == state]
            if state_gdf.empty:
                print(f"No geometry found for {state}")
                continue
            state_geoms[state] = ee.Geometry(state_gdf.geometry.iloc[0].__geo_interface__)

        comparative_maps = []
        for state in state_geoms:
            years = year_dict.get(state, ["2024"])
            if not isinstance(years, list):
                years = [years]
            years = sorted(set(str(y) for y in years if 2015 <= int(y) <= 2024))
            if len(years) < 2:
                print(f"Skipping comparative map for {state}: only {len(years)} year(s) available")
                continue

            for year in years:
                m = geemap.Map(zoom=7, height=400)
                boundary = ee.Feature(state_geoms[state], {"style": {"color": "black", "width": 2}})
                m.addLayer(boundary, {"style": "outline"}, f"{state} Boundary")

                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                s2_collection = None
                for attempt in range(3):
                    try:
                        s2_collection = (
                            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                            .filterBounds(state_geoms[state])
                            .filterDate(start_date, end_date)
                            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                            .sort("system:time_start", False)
                        )
                        size = s2_collection.size().getInfo()
                        if size > 0:
                            break
                        print(f"Attempt {attempt + 1}: No Sentinel-2 data for {state} {year}")
                        time.sleep(1)
                    except Exception as e:
                        if attempt == 2:
                            print(f"Failed to fetch Sentinel-2 for {state} {year}: {str(e)}")
                            s2_collection = None
                            break
                        time.sleep(1)

                if not s2_collection or s2_collection.size().getInfo() == 0:
                    print(f"No valid Sentinel-2 data for {state} {year}")
                    continue

                s2 = s2_collection.mosaic().clip(state_geoms[state]) if s2_collection.size().getInfo() > 1 else s2_collection.first().clip(state_geoms[state])

                if "NDVI" in requested_metrics:
                    ndvi = s2.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{state}_{year}")
                    m.addLayer(ndvi, {"min": 0, "max": 1, "palette": ["red", "yellow", "green"]}, f"NDVI ({state}, {year})")

                if "NBR" in requested_metrics:
                    nbr = s2.normalizedDifference(["B8", "B12"]).rename(f"NBR_{state}_{year}")
                    m.addLayer(nbr, {"min": -1, "max": 1, "palette": ["blue", "white", "red"]}, f"NBR ({state}, {year})")

                if "EVI" in requested_metrics:
                    nir = s2.select("B8")
                    red = s2.select("B4")
                    blue = s2.select("B2")
                    evi = nir.subtract(red).multiply(2.5).divide(
                        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
                    ).rename(f"EVI_{state}_{year}")
                    m.addLayer(evi, {"min": 0, "max": 1, "palette": ["red", "yellow", "green"]}, f"EVI ({state}, {year})")

                if "NDMI" in requested_metrics:
                    nir = s2.select("B8")
                    swir1 = s2.select("B11")
                    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename(f"NDMI_{state}_{year}")
                    m.addLayer(ndmi, {"min": -1, "max": 1, "palette": ["brown", "white", "blue"]}, f"NDMI ({state}, {year})")

                if "MNDWI" in requested_metrics:
                    green = s2.select("B3")
                    swir1 = s2.select("B11")
                    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename(f"MNDWI_{state}_{year}")
                    m.addLayer(mndwi, {"min": -1, "max": 1, "palette": ["brown", "white", "blue"]}, f"MNDWI ({state}, {year})")

                if any(metric in DYNAMIC_WORLD_CLASSES for metric in requested_metrics):
                    dw_collection = None
                    for attempt in range(3):
                        try:
                            dw_collection = (
                                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                                .filterBounds(state_geoms[state])
                                .filterDate(start_date, end_date)
                                .sort("system:time_start", False)
                            )
                            size = dw_collection.size().getInfo()
                            if size > 0:
                                break
                            print(f"Attempt {attempt + 1}: No Dynamic World data for {state} {year}")
                            time.sleep(1)
                        except Exception as e:
                            if attempt == 2:
                                print(f"Failed to fetch Dynamic World for {state} {year}: {str(e)}")
                                dw_collection = None
                                break
                            time.sleep(1)

                    if dw_collection and dw_collection.size().getInfo() > 0:
                        land_cover = dw_collection.mosaic().clip(state_geoms[state]).select("label").rename(f"Land_Cover_{state}_{year}")
                        land_cover_viz = {
                            "min": 0,
                            "max": 8,
                            "palette": ["419BDF", "397D49", "88B053", "7A87C6", "E49635", "DFC35A", "C4281B", "A59B8F", "B39FE1"]
                        }
                        m.addLayer(land_cover, land_cover_viz, f"Land Cover ({state}, {year})")
                    else:
                        print(f"No valid Dynamic World data for {state} {year}")

                m.centerObject(state_geoms[state], 7)
                comparative_maps.append({"state": state, "year": year, "map": m})
                print(f"Generated comparative map for {state} {year}")

        if comparative_maps:
            result_queue.put((comparative_maps, None))
        else:
            result_queue.put((None, "No comparative maps generated: insufficient years or data"))
    except Exception as e:
        result_queue.put((None, f"Comparative map generation failed: {str(e)}"))

# Streamlit UI Configuration
st.set_page_config(page_title="Environmental Data Explorer", layout="wide")

def get_theme_css(theme):
    if theme == "Dark":
        return """
            <style>
                body, .stApp {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                .chat-container {
                    min-height: 50px;
                    max-height: 60vh;
                    overflow-y: auto;
                    padding: 15px;
                    border: 1px solid #444;
                    border-radius: 8px;
                    background-color: #2b2b2b;
                    margin-bottom: 20px;
                }
                .user-msg {
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                    text-align: right;
                    max-width: 80%;
                    margin-left: auto;
                    color: #4a704a;
                }
                .bot-msg {
                    padding: 10px;
                    margin: 5px 0;
                    text-align: left;
                    max-width: 80%;
                    color: #ffffff;
                }
                .stChatInput {
                    position: fixed;
                    bottom: 20px;
                    width: 95%;
                    background-color: #333;
                    z-index: 1000;
                    padding: 10px;
                    color: #ffffff;
                    border: 1px solid #444;
                }
                footer {visibility: hidden;}
                .sidebar .sidebar-content {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                .stButton>button {
                    background-color: #444;
                    color: #ffffff;
                    border: 1px solid #666;
                    border-radius: 5px;
                }
                .stButton>button:hover {
                    background-color: #555;
                }
                .stSelectbox {
                    background-color: #333;
                    color: #ffffff;
                }
                .graph-container {
                    padding: 10px;
                    border-radius: 8px;
                    background-color: #2b2b2b;
                    margin: 5px;
                }
            </style>
        """
    else:
        return """
            <style>
                body, .stApp {
                    background-color: #ffffff;
                    color: #000000;
                }
                .user-msg {
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                    text-align: right;
                    max-width: 80%;
                    margin-left: auto;
                    color: #2e6b2e;
                }
                .bot-msg {
                    padding: 10px;
                    margin: 5px 0;
                    text-align: left;
                    max-width: 80%;
                    color: #000000;
                }
                footer {visibility: hidden;}
                .sidebar .sidebar-content {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                .stButton>button {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                .stButton>button:hover {
                    background-color: #d0d0d0;
                }
                .stSelectbox {
                    background-color: #ffffff;
                    color: #000000;
                }
                .graph-container {
                    padding: 10px;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                    margin: 5px;
                }
            </style>
        """

def main():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None
    if "theme" not in st.session_state:
        st.session_state.theme = "White"

    with st.sidebar:
        st.markdown("### Chat History")
        if st.session_state.chats:
            for chat_name in st.session_state.chats.keys():
                if st.button(chat_name, key=chat_name, use_container_width=True):
                    st.session_state.current_chat = chat_name
                    st.rerun()
        st.markdown("---")
        if st.button("New Chat", key="new_chat", use_container_width=True):
            st.session_state.current_chat = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Theme")
        theme = st.selectbox("Select Theme", ["White", "Dark"],
                            index=0 if st.session_state.theme == "White" else 1,
                            label_visibility="collapsed")
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

    st.title("🌍 Environmental Data Explorer")
    st.subheader("Analyze environmental metrics with graphs")

    if st.session_state.current_chat is None:
        messages = [{"role": "user", "content": "Ask about environmental data (e.g., 'Land cover for Kerala 2023')"}]
    else:
        messages = st.session_state.chats[st.session_state.current_chat]

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            if "report" in msg and msg["report"]:
                st.markdown("### Environmental Report")
                st.markdown(f'<div class="bot-msg">{msg["report"]}</div>', unsafe_allow_html=True)
            if "visualizations" in msg and msg["visualizations"]:
                st.markdown("### Visualizations")
                total_graphs = len(msg["visualizations"])
                for i in range(0, total_graphs, 2):
                    graphs_to_show = msg["visualizations"][i:i+2]
                    num_cols = len(graphs_to_show)
                    cols = st.columns(2) if num_cols == 2 else st.columns([1, 1])
                    for idx, fig in enumerate(graphs_to_show):
                        with cols[idx]:
                            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                            st.markdown(f"#### Visualization {i + idx + 1}")
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
            if "map" in msg and msg["map"]:
                st.markdown("### GEE Map")
                with st.spinner("Loading GEE Map..."):
                    msg["map"].to_streamlit(height=400)
            if "comparative_maps" in msg and msg["comparative_maps"]:
                st.markdown("### Comparative GEE Maps Across Years")
                for state in set(m["state"] for m in msg["comparative_maps"]):
                    state_maps = [m for m in msg["comparative_maps"] if m["state"] == state]
                    if len(state_maps) > 1:
                        st.markdown(f"#### {state}")
                        tabs = st.tabs([f"{m['year']}" for m in state_maps])
                        for tab, map_data in zip(tabs, state_maps):
                            with tab:
                                with st.spinner(f"Loading map for {state} {map_data['year']}..."):
                                    map_data["map"].to_streamlit(height=400)
                                st.caption(f"Land Cover and Metrics for {state} ({map_data['year']})")
            if "error" in msg:
                st.error(msg["error"])
    st.markdown("</div>", unsafe_allow_html=True)

    query = st.chat_input("Type your query here (e.g., 'Land cover for Kerala 2023')")
    if query:
        if st.session_state.current_chat is None:
            chat_name = query[:50]
            if chat_name in st.session_state.chats:
                chat_name += f"_{len(st.session_state.chats)}"
            st.session_state.chats[chat_name] = [
                {"role": "user", "content": "Ask about environmental data (e.g., 'Land cover for Kerala 2023')"}
            ]
            st.session_state.current_chat = chat_name

        messages = st.session_state.chats[st.session_state.current_chat]
        messages.append({"role": "user", "content": query})
        
        detected_states = extract_states_from_query(query)
        if not detected_states:
            messages.append({"role": "assistant", "error": "Please include a state name."})
        else:
            year_dict = extract_year(query)
            requested_metrics = extract_metrics_from_query(query)
            checkout_corpus_data = ""
            for state in detected_states:
                corpus_file_path = os.path.join(CORPUS_FOLDER, state_corpus_files[state])
                if os.path.exists(corpus_file_path):
                    with open(corpus_file_path, "r", encoding="utf-8") as f:
                        checkout_corpus_data += f"\n--- {state} ---\n" + f.read()
                else:
                    messages.append({"role": "assistant",
                                   "error": f"No data file for {state} at {corpus_file_path}."})
                    st.rerun()

            with st.spinner("Processing your request..."):
                mistral_values = call_mistral_saba(MISTRAL_API_URL, MISTRAL_API_KEY,
                                                 checkout_corpus_data, query,
                                                 detected_states, requested_metrics)
                print(f"Mistral Values: {mistral_values}")

                response = {"role": "assistant"}
                if "API Error" in mistral_values:
                    response["error"] = mistral_values
                else:
                    # Generate report
                    report = generate_report(query, detected_states, year_dict, checkout_corpus_data, mistral_values)
                    response["report"] = report

                    # Generate visualizations
                    viz_figs = generate_visualization(mistral_values, detected_states,
                                                    year_dict, query, requested_metrics)
                    if viz_figs:
                        response["visualizations"] = viz_figs
                    else:
                        response["error"] = "No visualizations generated. Check debug logs for details: Mistral Values = " + mistral_values

                    # Generate GEE maps (original and comparative)
                    map_result_queue = Queue()
                    comp_map_result_queue = Queue()

                    map_thread = threading.Thread(target=generate_map, args=(detected_states, year_dict, query, map_result_queue))
                    map_thread.start()

                    has_multiple_years = any(len(year_dict[state]) > 1 for state in detected_states)
                    if has_multiple_years:
                        comp_map_thread = threading.Thread(target=generate_comparative_maps, args=(detected_states, year_dict, query, requested_metrics, comp_map_result_queue))
                        comp_map_thread.start()
                        comp_map_thread.join()

                    map_thread.join()

                    map_obj, map_error = map_result_queue.get()
                    if map_error:
                        response["error"] = map_error
                    elif map_obj:
                        response["map"] = map_obj

                    if has_multiple_years:
                        comp_maps, comp_map_error = comp_map_result_queue.get()
                        if comp_map_error:
                            response["error"] = comp_map_error
                        elif comp_maps:
                            response["comparative_maps"] = comp_maps

                messages.append(response)
                st.session_state.chats[st.session_state.current_chat] = messages
                st.rerun()

if __name__ == "__main__":
    main()
