import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def create_model():
    print("Crééation des fichiers modèle...")

    try:
        df = pd.read_csv('../fichier_nettoye50000.csv')
        print(f"Données chargées: {df.shape}")
    except:
        print("Fichier non trouvé, création de données d'exemple baséessur votre structure...")
        df = create_sample_data()

    df_clean = clean_data(df)
    df_features = extract_features(df_clean)
    features = df_features.drop('Accident_Severity', axis=1)
    target = df_features['Accident_Severity']

    label_encoder = {}
    categorical_cols = features.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        label_encoder[col] = LabelEncoder()
        features[col] = label_encoder[col].fit_transform(features[col])

    scaler = StandardScaler()
    numerical_cols = features.select_dtypes(include=[np.number]).columns
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print(f"Score d'entrainement: {train_score:.2%}")
    print(f"Score test: {test_score:.2%}")
    print(f"Classes {model.classes_}")

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, "model/accident_model.pkl")
    jolib.dump({
        'scaler': scaler,
        'label_encoder': label_encoder},
        "model/accident_scaler.pkl"
    )

    print('Modele créé et sauvegardé')

def clean_data(df):
    df_clean = df.copy()

    df_clean.fillna({
        'Junction_Control': 'Unknown',
        'Junction_Detail': 'Unknown',
        'Road_Surface_Conditions': 'Unknown',
        'Weather_Conditions': 'Unknown'
    }, inplace=True)

    df_clean["Number_of_Casualties"] = pd.to_numeric(df_clean["Number_of_Casualties"], errors='coerce').fillna(1)
    df_clean["Speed_limit"] = pd.to_numeric(df_clean["Speed_limit"], errors='coerce').fillna(30)

    return df_clean

def extract_features(df):
    df_features = df.copy()

    df_features["Hour"] = df_features["Time"].str.split(":")[0].fillna('12').astype(int)
    df_features["Month"] = df_features["Accident_Date"].str.split("/")[0].fillna('6').astype(int)
    df_features["Is_Weekend"] = df_features["Day_of_Week"].isin(['Sunday', 'Saturday']).astype(int)

    return df_features


def create_sample_data():
    np.random.seed(42)
    n_samples = 1000

    data = {
        'Accident_Date': [f"{np.random.randint(1, 13)}/{np.random.randint(1, 29)}/2022" for _ in range(n_samples)],
        'Day_of_Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples),
        'Junction_Control': np.random.choice(['Give way or uncontrolled', 'Auto traffic signal', 'Stop sign', 'Unknown'], n_samples),
        'Junction_Detail': np.random.choice(['T or staggered junction', 'Crossroads', 'Roundabout', 'Not at junction', 'Unknown'], n_samples),
        'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples, p=[0.7, 0.25, 0.05]),
        'Light_Conditions': np.random.choice(['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit'], n_samples),
        'Number_of_Casualties': np.random.randint(1, 5, n_samples),
        'Road_Surface_Conditions': np.random.choice(['Dry', 'Wet/Damp', 'Frost/Ice', 'Snow', 'Flood over 3cm. deep'], n_samples),
        'Road_Type': np.random.choice(['Single carriageway', 'Dual carriageway', 'Roundabout', 'One way street'], n_samples),
        'Speed_limit': np.random.choice([30, 50, 60, 70, 90], n_samples),
        'Time': [f"{np.random.randint(0, 24)}:{np.random.randint(0, 60):02d}" for _ in range(n_samples)],
        'Urban_or_Rural_Area': np.random.choice(['Urban', 'Rural'], n_samples),
        'Weather_Conditions': np.random.choice(['Fine no high winds', 'Raining no high winds', 'Fog or mist', 'Snowing no high winds', 'Unknown'], n_samples)
    }

    return pd.DataFrame(data)

if __name__ == "__main__":
    create_model()