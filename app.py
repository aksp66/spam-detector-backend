# app.py - Backend FastAPI pour le détecteur de spam

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import requests
import io

# ============================================
# CONFIGURATION DE L'APPLICATION
# ============================================

app = FastAPI(title="Détecteur de Spam SMS", version="1.0.0")

# Configuration CORS pour permettre au frontend React de communiquer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez l'URL exacte de votre frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODÈLES DE DONNÉES (pour l'API)
# ============================================

class Message(BaseModel):
    """Structure des données envoyées par le frontend"""
    text: str

class PredictionResponse(BaseModel):
    """Structure de la réponse de prédiction"""
    text: str
    prediction: str  # "spam" ou "ham"
    confidence: float  # Score de confiance (0-100%)

# ============================================
# VARIABLES GLOBALES (modèle et vectoriseur)
# ============================================

model = None
vectorizer = None
model_stats = {}

# ============================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ============================================

def load_dataset():
    """
    Charge le dataset SMS Spam Collection depuis UCI
    Format : label\tmessage
    """
    print("📥 Téléchargement du dataset...")
    
    # URL du dataset SMS Spam Collection
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        # Téléchargement
        response = requests.get(url)
        response.raise_for_status()
        
        # Extraction du fichier ZIP
        import zipfile
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Le fichier s'appelle 'SMSSpamCollection'
            with zip_file.open('SMSSpamCollection') as f:
                # Lecture avec pandas (séparateur = tabulation)
                df = pd.read_csv(f, sep='\t', names=['label', 'message'], encoding='latin-1')
        
        print(f"✅ Dataset chargé : {len(df)} messages")
        print(f"   - Ham (non-spam) : {len(df[df['label']=='ham'])}")
        print(f"   - Spam : {len(df[df['label']=='spam'])}")
        
        return df
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        # Dataset de secours pour les tests
        return pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam'],
            'message': [
                'Hello, how are you?',
                'FREE! Click here to win $1000 now!!!',
                'See you tomorrow at the meeting',
                'Congratulations! You have won a free iPhone. Call now!'
            ]
        })

# ============================================
# ÉTAPE 2 : PRÉTRAITEMENT
# ============================================

def preprocess_data(df):
    """
    Nettoie et prépare les données
    - Convertit les labels en 0/1
    - Vérifie l'absence de valeurs manquantes
    """
    print("🧹 Prétraitement des données...")
    
    # Conversion des labels : ham=0, spam=1
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Suppression des éventuelles lignes vides
    df = df.dropna()
    
    print(f"✅ Prétraitement terminé : {len(df)} messages prêts")
    return df

# ============================================
# ÉTAPE 3 : VECTORISATION (TF-IDF)
# ============================================

def create_features(X_train, X_test):
    """
    Convertit les textes en vecteurs numériques avec TF-IDF
    
    TF-IDF = Term Frequency - Inverse Document Frequency
    - Donne plus de poids aux mots rares et discriminants
    - Réduit l'importance des mots très fréquents
    """
    print("🔢 Vectorisation TF-IDF...")
    
    # Création du vectoriseur
    global vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,           # Convertit en minuscules
        stop_words='english',     # Supprime les mots courants (the, is, at...)
        max_features=3000,        # Garde les 3000 mots les plus importants
        ngram_range=(1, 2)        # Utilise les mots seuls et les paires de mots
    )
    
    # Apprentissage et transformation sur les données d'entraînement
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Transformation (sans ré-apprentissage) sur les données de test
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ Vectorisation terminée : {X_train_vec.shape[1]} features")
    return X_train_vec, X_test_vec

# ============================================
# ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE
# ============================================

def train_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle Multinomial Naïve Bayes
    
    Pourquoi Naïve Bayes ?
    - Très efficace sur les données textuelles
    - Rapide à entraîner
    - Performant même avec peu de données
    """
    print("🤖 Entraînement du modèle Naïve Bayes...")
    
    global model, model_stats
    
    # Création du modèle
    model = MultinomialNB(alpha=0.1)  # alpha = paramètre de lissage
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions sur le jeu de test
    y_pred = model.predict(X_test)
    
    # Calcul des métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    
    # Sauvegarde des statistiques
    model_stats = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    print(f"✅ Modèle entraîné avec succès !")
    print(f"   Précision globale : {accuracy*100:.2f}%")
    print("\n📊 Rapport de classification :")
    print(class_report)
    
    return model

# ============================================
# ÉTAPE 5 : SAUVEGARDE DU MODÈLE
# ============================================

def save_model():
    """Sauvegarde le modèle et le vectoriseur pour réutilisation"""
    print("💾 Sauvegarde du modèle...")
    
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✅ Modèle sauvegardé")

def load_model():
    """Charge le modèle sauvegardé (si disponible)"""
    global model, vectorizer
    
    try:
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("✅ Modèle chargé depuis les fichiers")
        return True
    except FileNotFoundError:
        print("⚠️ Aucun modèle sauvegardé trouvé")
        return False

# ============================================
# INITIALISATION AU DÉMARRAGE
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Fonction exécutée au démarrage de l'application
    - Tente de charger un modèle existant
    - Sinon, entraîne un nouveau modèle
    """
    print("\n" + "="*50)
    print("🚀 DÉMARRAGE DU DÉTECTEUR DE SPAM")
    print("="*50 + "\n")
    
    # Essai de chargement d'un modèle existant
    if not load_model():
        # Si pas de modèle, on en entraîne un nouveau
        print("🔄 Entraînement d'un nouveau modèle...")
        
        # 1. Chargement des données
        df = load_dataset()
        
        # 2. Prétraitement
        df = preprocess_data(df)
        
        # 3. Séparation train/test (80% / 20%)
        X = df['message']
        y = df['label_num']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Données divisées :")
        print(f"   - Entraînement : {len(X_train)} messages")
        print(f"   - Test : {len(X_test)} messages\n")
        
        # 4. Vectorisation
        X_train_vec, X_test_vec = create_features(X_train, X_test)
        
        # 5. Entraînement
        train_model(X_train_vec, y_train, X_test_vec, y_test)
        
        # 6. Sauvegarde
        save_model()
    
    print("\n" + "="*50)
    print("✅ DÉTECTEUR DE SPAM PRÊT !")
    print("="*50 + "\n")

# ============================================
# ROUTES DE L'API
# ============================================

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Détecteur de Spam SMS - API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "stats": "/stats (GET)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifie que l'API fonctionne"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.get("/stats")
async def get_stats():
    """Retourne les statistiques du modèle"""
    if not model_stats:
        raise HTTPException(status_code=503, detail="Modèle non entraîné")
    
    return {
        "accuracy": f"{model_stats['accuracy']*100:.2f}%",
        "confusion_matrix": model_stats['confusion_matrix'],
        "details": "Matrice de confusion : [[True Ham, False Spam], [False Ham, True Spam]]"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(message: Message):
    """
    Endpoint principal : prédit si un message est spam ou ham
    
    Paramètres :
    - message.text : Le texte du message à analyser
    
    Retour :
    - prediction : "spam" ou "ham"
    - confidence : Score de confiance en %
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non disponible. Veuillez réessayer dans quelques instants."
        )
    
    try:
        # 1. Vectorisation du message
        message_vec = vectorizer.transform([message.text])
        
        # 2. Prédiction
        prediction = model.predict(message_vec)[0]
        
        # 3. Calcul de la confiance (probabilité)
        proba = model.predict_proba(message_vec)[0]
        confidence = max(proba) * 100  # Confiance en %
        
        # 4. Conversion du résultat
        label = "spam" if prediction == 1 else "ham"
        
        return PredictionResponse(
            text=message.text,
            prediction=label,
            confidence=round(confidence, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

# ============================================
# LANCEMENT DE L'APPLICATION
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    # Lancement du serveur
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Écoute sur toutes les interfaces
        port=8000,
        reload=True      # Rechargement automatique en développement
    )