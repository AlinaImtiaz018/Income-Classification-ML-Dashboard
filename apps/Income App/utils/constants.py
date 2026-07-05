MODEL_NAMES_SHORT = ["Logistic Reg.", "Random Forest", "Grad. Boosting", "SVM"]
MODEL_NAMES_FULL = [
    "Linear Classification",
    "Random Forest Classification",
    "Gradient Boosting Classification",
    "Support Vector Machines (SVMs)",
]
COLUMNS = [
    "Age", "Employment_type", "Weighting_factor", "Education_level",
    "Schooling_years", "Marital_status", "Employment_area", "Partnership",
    "Ethnicity", "Gender", "Gains", "Losses", "Weekly_working_time",
    "Country_of_birth", "Income",
]
NUM_FEATURES = ["Age", "Weighting_factor", "Schooling_years", "Gains", "Losses", "Weekly_working_time"]
CAT_FEATURES = ["Employment_type", "Education_level", "Marital_status", "Employment_area",
                "Partnership", "Ethnicity", "Gender", "Country_of_birth"]
RESULTS = {
    "KNN": {
        "Linear Classification":            {"accuracy": 0.806, "f1_score": 0.503, "recall": 0.415, "precision": 0.636, "roc_auc": 0.671, "cm": [[354,28],[69,49]]},
        "Random Forest Classification":     {"accuracy": 0.844, "f1_score": 0.639, "recall": 0.585, "precision": 0.704, "roc_auc": 0.754, "cm": [[353,29],[49,69]]},
        "Gradient Boosting Classification": {"accuracy": 0.870, "f1_score": 0.680, "recall": 0.585, "precision": 0.812, "roc_auc": 0.771, "cm": [[366,16],[49,69]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.772, "f1_score": 0.066, "recall": 0.034, "precision": 1.000, "roc_auc": 0.517, "cm": [[382,0],[114,4]]},
    },
    "Iterative": {
        "Linear Classification":            {"accuracy": 0.806, "f1_score": 0.481, "recall": 0.381, "precision": 0.652, "roc_auc": 0.659, "cm": [[358,24],[73,45]]},
        "Random Forest Classification":     {"accuracy": 0.842, "f1_score": 0.636, "recall": 0.585, "precision": 0.697, "roc_auc": 0.753, "cm": [[352,30],[49,69]]},
        "Gradient Boosting Classification": {"accuracy": 0.868, "f1_score": 0.680, "recall": 0.593, "precision": 0.795, "roc_auc": 0.773, "cm": [[364,18],[48,70]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.772, "f1_score": 0.066, "recall": 0.034, "precision": 1.000, "roc_auc": 0.517, "cm": [[382,0],[114,4]]},
    },
    "Dropped": {
        "Linear Classification":            {"accuracy": 0.814, "f1_score": 0.465, "recall": 0.343, "precision": 0.725, "roc_auc": 0.651, "cm": [[336,14],[71,37]]},
        "Random Forest Classification":     {"accuracy": 0.849, "f1_score": 0.657, "recall": 0.611, "precision": 0.710, "roc_auc": 0.767, "cm": [[323,27],[42,66]]},
        "Gradient Boosting Classification": {"accuracy": 0.871, "f1_score": 0.694, "recall": 0.620, "precision": 0.788, "roc_auc": 0.784, "cm": [[332,18],[41,67]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.769, "f1_score": 0.036, "recall": 0.019, "precision": 1.000, "roc_auc": 0.509, "cm": [[350,0],[106,2]]},
    },
}
PREDICTED_COUNTS = {
    "KNN":       {"<=50K": 20324, ">50K": 4676},
    "Iterative": {"<=50K": 20955, ">50K": 4045},
    "Dropped":   {"<=50K": 18812, ">50K": 4412},
}
