""" routes.py - Flask route definitions

Flask requires routes to be defined to know what data to provide for a given
URL. The routes provided are relative to the base hostname of the website, and
must begin with a slash."""
from flaskapp import app
from flask import render_template, request, jsonify
import json
import numpy as np
import random

from search import TextSearch

text_search = TextSearch("ga_final.parquet", "BOW")

@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/network', methods=["GET", "POST"])
def network():
    return render_template('network.html')

@app.route('/map', methods=["GET", "POST"])
def map():
    return render_template('map.html')

@app.route('/data')
def data():
    with open("temp.json", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

@app.route('/get_case_graph/<case_id>')
def get_case_graph(case_id):
    # print(search_results)
    ret = {"cases" : [], "links" : []}
    
    
    root_row = text_search.df[text_search.df['id'] == case_id].iloc[0]
    cases = []
    links = []

    # Need to add a check to make sure there are no repeated nodes
    seen_ids = set()
    seen_ids.add(case_id) 

    cases.append(
        {
            "id" : root_row["id"],
            "name" : root_row["name"],
            "head_matter" : root_row["head_matter"],
            "decision_date" : root_row["decision_date"],
            "name_abbreviation" : root_row["name_abbreviation"],
            "jurisdiction" : root_row["jurisdiction"],
            "text" : root_row["text"],
            "case_type" : root_row["category"] if root_row["category"] else "Unknown",
        }
    )
    
    old_rows = [root_row]
    new_rows = []
    for _ in range(4):
        for iter_row in old_rows:
            row_text = iter_row['text']
            search_results = text_search.search(row_text, 3)
            for index, row in search_results.iterrows():
                if {"source" : row["id"], "target" : iter_row["id"]} not in links:
                    links.append({"source" : iter_row["id"], "target" : row["id"]})

                if row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                new_rows.append(row)
                cases.append(
                    {
                        "id" : row["id"],
                        "name" : row["name"],
                        "head_matter" : row["head_matter"],
                        "decision_date" : row["decision_date"],
                        "name_abbreviation" : row["name_abbreviation"],
                        "jurisdiction" : row["jurisdiction"],
                        "text" : row["text"],
                        "case_type" : row["category"] if row["category"] else "Unknown",
                    }
                )
                
        old_rows = new_rows
        new_rows = []

    ret["cases"] = cases
    ret["links"] = links
    return ret, 200

@app.route('/get_root_cases')
def get_root_cases():
    print(len(text_search.df))

    ret = {"cases" : []}
    cases = []
    sampled_integers = np.random.choice(len(text_search.df) + 1, size=10, replace=False)
    for i in sampled_integers:
        cases.append(
            {
                "id" : text_search.df["id"][i],
                "name" : text_search.df["name"][i],
                "head_matter" : text_search.df["head_matter"][i]
            }
        )

    ret["cases"] = cases
    return ret, 200
