
# Comic Signal Engine

## Overview

Comic Signal Engine is a machine learning research project that explores which characteristics of a comic book contribute to its long-term collectible value.

The goal is to determine whether measurable attributes—such as publisher, characters, release year, story information, and other metadata—can reveal patterns associated with comics that appreciate over time.

This project began with a simple question:

**Can machine learning identify collectible signals that human collectors might overlook?**

Rather than relying solely on opinions or speculation, Comic Signal Engine is being developed to analyze historical data, discover meaningful relationships, and eventually predict whether a newly released comic is likely to become highly sought after.

The long-term vision is to build an intelligent decision-support tool that helps collectors make more informed purchasing decisions while reducing the risk of investing in comics that never gain significant value.

## Current Status

This project is an active prototype.

The current version performs feature engineering and clustering on comic metadata to identify patterns associated with collectible potential.

Future versions will incorporate historical market pricing, sales data, and supervised machine learning models to predict long-term appreciation.

## Project Structure

```
build_artifacts.py    Trains the clustering model and generates artifacts
scoring.py            Loads the trained model and performs predictions
app.py                Flask web application
Marvel_Comics.csv     Source dataset
model_bundle.joblib   Saved machine learning model
```

Run:
1. python build_artifacts.py
2. python app.py
3. open http://127.0.0.1:5000