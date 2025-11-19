import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def create_model():
    print("création des fichiers modèle...")

    try:
        df = pd.read_csv('../fichier_nettoye50000.csv', sep=',')
        print(f"données chargées: {df.shape}")
    except Exception as e:
        print(f"fichier non trouvé ({e}), création de données d'exemple...")
        df = create_sample_data()

    df_clean = clean_data(df)

    colonnes_a_supprimer = ['number_of_casualties', 'time', 'accident_severity', 'accident_date']
    df_clean = df_clean.drop(columns=[col for col in colonnes_a_supprimer if col in df_clean.columns])

    if df_clean.empty:
        print(" aucune colonne restante après suppression")
        return

    print(f"colonnes utilisées: {list(df_clean.columns)}")

    print("ORDRE DES COLONNES POUR L'ENTRAÎNEMENT:")
    for i, col in enumerate(df_clean.columns[:-1]):
        print(f"  {i + 1}. {col}")
    print(f"  Target: {df_clean.columns[-1]}")

    features = df_clean.iloc[:, :-1]
    target = df_clean.iloc[:, -1]

    label_encoder = {}
    categorical_cols = features.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        label_encoder[col] = LabelEncoder()
        features[col] = label_encoder[col].fit_transform(features[col])

    scaler = StandardScaler()
    numerical_cols = features.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print(f"score d'entrainement: {train_score:.2%}")
    print(f"score test: {test_score:.2%}")
    print(f"classes: {model.classes_}")

    os.makedirs('model', exist_ok=True)

    joblib.dump(model, "model/accident_model.pkl")

    joblib.dump({
        'scaler': scaler,
        'label_encoder': label_encoder
    }, "model/accident_scaler.pkl")

    print('modèle créé et sauvegardé avec succès!')


def clean_data(df):
    df_clean = df.copy()

    df_clean.fillna({
        'junction_control': 'unknown',
        'junction_detail': 'unknown',
        'road_surface_conditions': 'unknown',
        'weather_conditions': 'unknown'
    }, inplace=True)

    if 'speed_limit' in df_clean.columns:
        df_clean["speed_limit"] = pd.to_numeric(df_clean["speed_limit"], errors='coerce').fillna(30)

    return df_clean


def create_sample_data():
    np.random.seed(42)
    n_samples = 1000

    data = {
        'day_of_week': np.random.choice(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], n_samples),
        'junction_control': np.random.choice(['give way or uncontrolled', 'auto traffic signal', 'stop sign', 'unknown'], n_samples),
        'junction_detail': np.random.choice(['t or staggered junction', 'crossroads', 'roundabout', 'not at junction', 'unknown'], n_samples),
        'light_conditions': np.random.choice(['daylight', 'darkness - lights lit', 'darkness - no lighting', 'darkness - lights unlit'], n_samples),
        'road_surface_conditions': np.random.choice(['dry', 'wet/damp', 'frost/ice', 'snow', 'flood over 3cm. deep'], n_samples),
        'road_type': np.random.choice(['single carriageway', 'dual carriageway', 'roundabout', 'one way street'], n_samples),
        'speed_limit': np.random.choice([30, 50, 60, 70, 90], n_samples),
        'urban_or_rural_area': np.random.choice(['urban', 'rural'], n_samples),
        'weather_conditions': np.random.choice(['fine no high winds', 'raining no high winds', 'fog or mist', 'snowing no high winds', 'unknown'], n_samples),
        'gravite': np.random.choice(['leger', 'grave', 'mortel'], n_samples, p=[0.7, 0.25, 0.05])
    }

    return pd.DataFrame(data)

if __name__ == "__main__":
    create_model()