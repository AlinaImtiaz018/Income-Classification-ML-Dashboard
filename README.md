# Machine Learning - Income Classification Streamlit Demonstrator

This repository contains an interactive **Streamlit application** that demonstrates the functionality of my machine learning income project. This application provides a user-friendly interface for exploring the dataset, running trained models, visualizing results, and understanding the underlying machine learning workflow.

The purpose of this repository is to make my project accessible through an interactive web application without requiring users to execute notebooks or write code.

---

## Features

* Interactive machine learning demonstrator
* Data exploration and visualization
* Model prediction interfaces
* Performance evaluation dashboard
* Downloadable results (where applicable)
* Responsive Streamlit user interface

---

## Project Structure

```text
Income Classification ML Dashboard/
|
├── .streamlit/
|   └── config.toml
├── assets/
|   └── style.css
├── data/
|   └── einkommen.train
├── pages_modules/
|   ├── data_exploration.py
|   ├── model_comparison.py
|   ├── model_results.py
|   ├── overview.py
|   └── predictions.py
├── utils/
|   ├── colors.py
|   ├── constants.py
|   ├── data.py
|   ├── layout.py
|   ├── sidebar.py
|   └── styles.py
├── CITATION.cff
├── CONDUCT.md
├── LICENSE
├── README.md
├── income-app.py
└── requirements.txt
```

The Streamlit application demonstrates my machine learning project.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AlinaImtiaz018/Income-Classification-ML-Dashboard.git
cd Income-Classification-ML-Dashboard
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Applications

Launch Streamlit:

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Plotly
* Matplotlib

---

## Purpose

This repository serves as a portfolio of the interactive demonstration for my machine learning income project, making it easy to explore model behavior, predictions, and evaluation results through a web interface.

---

## Citation

If you use this repository in your work or research, please cite it using the provided **CITATION.cff** file.

---

## License

This project is licensed under the MIT License. See the **LICENSE** file for details.
