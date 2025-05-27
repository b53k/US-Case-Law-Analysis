# US Case Law Visualization with NLP and Machine Learning

Legal research is a critical part of the justice system. Lawyers and judges rely on past cases to make decisions — but reading through millions of legal documents is slow, manual, and inefficient. With over 6.7 million U.S. court cases available, traditional keyword searches fall short. They miss deeper patterns, relationships, and trends hidden in complex legal texts.

The goal of this project is to make legal research smarter, faster, and more interactive using data analytics and visualization. We developed a system that helps users explore case law with natural language processing (NLP), network graphs, and predictive modeling — turning text into insights.

## Installing the data

Some of the raw data is too large to be zipped up and submitted. We need to install it in order to do the processing on the data to be used in the visualizations. Use the following links to access the data and install it to the root of the `CODE` directory of this project.

[Georgia Dataset](https://gtvault.sharepoint.com/:u:/s/CSE6242-GroupProject477/ETV7csPe5glAoWen6Aubg0IBARlye1kXNMP4szWzy9aq-A?e=Eodrnw)

## Environment

All of the dependencies for this project are packaged inside of a Python virtual environment. To activate the environment run the following if you are on a Window's PC. Note that you may need to enable Powershell scripts to be run as a user to activate the environment. Also note that your Python interpreter could be named `python` or `python3`.

```powershell
python -m venv venv
./venv/Scripts/Activate
pip install -r requirements.txt
```

If you are on a unix based system run the following to activate the environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Cleaning and feature engineering

The data is already cleaned and there is a feature engineering process to create the complexity, case type, and subtype columns of the dataset. This was accomplished by using `scripts/caseCategorization.py`. This process does not need to be replicated, and the original source data is not provided for this demonstration, and could be gotten from HuggingFace if access is granted by maintainers.

## Run the word embedding preprocessing

We need to create word embeddings of the Georgia dataset that we installed earlier so that we can use a K-nearest neighbors algorithm in order to create a network visualization of the cases. In order to do this run `python ./preprocess.py ga_final.parquet BOW` in a terminal. After doing this, you will have created several files related to the BOW embeddings of the data set. You can create additional embeddings using different positional arguments in the preprocess command, but only the BOW embeddings will be used for the visualization. The additional embeddings are: bge, legalbert, TFIDF, and doc2vec.

To create the evaluation results of the different word embeddings `eval.py` can be used to get the results seen in the final report and on the poster.

## Run preprocessing for the choropleth map

The data for the processed Choropleth is already provided, but in the case of wanting to reprocess this data like we had, `scripts/case_counts.py` would be used.

## Launch the application

After prepping all the data you are ready to run the main application and see the outputted graph. Run `python run.py` from the root directory. After this, the website will start running and serve the home page at `localhost:3001` in your web browser.

## 🗺️ US Case Law Visualization Dashboard

This interactive dashboard visualizes historical case law data across US states using a choropleth map and a time-series density plot. It allows users to explore the **volume of legal cases by year and state** from 1658 to 2019. Additionally there is an interactive network graph visualization tool for the Georgia Case Law.

---

### 📊 Features

- **Choropleth Map** of the United States showing total case count per state for a selected year.
- **Interactive Year Input** to dynamically update map values.
- **Density Plot** below the map showing how case counts are distributed over time for a selected state.
- **Smooth Kernel Density Estimation (KDE)** using the Epanechnikov kernel for accurate visual trends.
- **Tooltips**, **color legend**, and dropdown menus for intuitive user experience.
- **Network graph** Generate a graph of Georgia Case Law.

---

### 📷 Dashboard

#### Choropleth Map

![Visualization](./flaskapp/static/Visualize.png)

#### Network Graph

![Visualization](./flaskapp/static/network_graph.png)

### 📂 Dataset

The dataset includes over 6 million US case records from the [Caselaw Access Project](https://case.law/) and is preprocessed into a CSV format with:

```csv
Jurisdiction,Year,Case Count
California,1962,1622
Maine,1910,140
Washington,1949,299
```

The Georgia Case Law parquet file was installed earlier and contains all of the cleaned data for the Network Visualization tool.
