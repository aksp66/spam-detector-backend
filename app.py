# ============================================
# APP.PY - DÉTECTEUR DE SPAM HYBRIDE INTELLIGENT
# ============================================
# Architecture : Logistic Regression + Analyse de patterns
# Niveau 1 : Filtre rapide ML (0.1 sec)
# Niveau 2 : Analyse approfondie des patterns dangereux
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import requests
import io
import zipfile

# ============================================
# CONFIGURATION DE L'APPLICATION
# ============================================

app = FastAPI(
    title="Détecteur de Spam Hybride",
    description="IA intelligente combinant ML et analyse de patterns",
    version="2.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODÈLES DE DONNÉES
# ============================================

class Message(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    method: str  # "ml_fast" ou "ml_deep_analysis"
    danger_signals: list  # Liste des signaux de danger détectés

# ============================================
# VARIABLES GLOBALES
# ============================================

model = None
vectorizer = None
model_stats = {}

# ============================================
# ANALYSEUR DE PATTERNS DANGEREUX
# ============================================
# Cette classe détecte des signaux de spam au-delà des mots

class DangerPatternAnalyzer:
    """
    Analyse approfondie des patterns de spam
    Détecte : URLs suspectes, urgence, MAJUSCULES, argent, numéros
    """
    
    def __init__(self):
        # Mots-clés d'urgence (psychologie de la pression)
        self.urgency_words = [
            'urgent', 'now', 'immediately', 'hurry', 'limited time',
            'expire', 'last chance', 'act now', 'don\'t wait',
            'urgent', 'maintenant', 'immédiatement', 'vite', 'limité'
        ]
        
        # Mots-clés d'argent (appât financier)
        self.money_words = [
            'free', 'win', 'winner', 'cash', 'prize', 'million',
            'dollars', 'euros', '$', '€', 'money', 'rich', 'earn',
            'gratuit', 'gagner', 'gagnant', 'prix', 'argent'
        ]
        
        # Mots de demande d'action (phishing)
        self.action_words = [
            'click', 'call', 'verify', 'confirm', 'update', 'download',
            'install', 'register', 'claim', 'redeem',
            'cliquer', 'appeler', 'vérifier', 'confirmer', 'télécharger'
        ]
    
    def analyze(self, text):
        """
        Analyse complète d'un message
        Retourne : score de danger (0-100) et liste des signaux
        """
        text_lower = text.lower()
        danger_score = 0
        signals = []
        
        # 1. DÉTECTION D'URLs (les spams contiennent souvent des liens)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        shortened_urls = re.findall(r'\b(?:bit\.ly|tinyurl|goo\.gl|ow\.ly|t\.co)/\w+', text_lower)
        
        if urls or shortened_urls:
            danger_score += 20
            signals.append(f"🔗 Contient {len(urls) + len(shortened_urls)} URL(s)")
        
        # 2. DÉTECTION DE NUMÉROS DE TÉLÉPHONE
        # Formats : +33, 06, 07, (555) 123-4567, etc.
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\b0[6-7]\d{8}\b',  # Français
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # US
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        if phones:
            danger_score += 15
            signals.append(f"📞 Contient {len(phones)} numéro(s) de téléphone")
        
        # 3. DÉTECTION DE MAJUSCULES EXCESSIVES
        # Les spammeurs crient pour attirer l'attention
        if text.isupper() or len([c for c in text if c.isupper()]) / max(len(text), 1) > 0.5:
            danger_score += 25
            signals.append("🔊 TEXTE EN MAJUSCULES (crie pour attirer l'attention)")
        
        # 4. DÉTECTION DE POINTS D'EXCLAMATION MULTIPLES
        exclamations = text.count('!')
        if exclamations >= 3:
            danger_score += 15
            signals.append(f"❗ {exclamations} points d'exclamation (urgence artificielle)")
        
        # 5. DÉTECTION DE MOTS D'URGENCE
        urgency_count = sum(1 for word in self.urgency_words if word in text_lower)
        if urgency_count > 0:
            danger_score += urgency_count * 10
            signals.append(f"⏰ {urgency_count} mot(s) d'urgence détecté(s)")
        
        # 6. DÉTECTION DE MOTS D'ARGENT
        money_count = sum(1 for word in self.money_words if word in text_lower)
        if money_count > 0:
            danger_score += money_count * 8
            signals.append(f"💰 {money_count} mot(s) lié(s) à l'argent")
        
        # 7. DÉTECTION DE DEMANDES D'ACTION
        action_count = sum(1 for word in self.action_words if word in text_lower)
        if action_count > 0:
            danger_score += action_count * 7
            signals.append(f"👆 {action_count} demande(s) d'action")
        
        # 8. DÉTECTION D'EMAILS SUSPECTS
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            danger_score += 10
            signals.append(f"📧 Contient {len(emails)} adresse(s) email")
        
        # 9. DÉTECTION DE SYMBOLES MONÉTAIRES
        currency_symbols = len(re.findall(r'[$€£¥₹]', text))
        if currency_symbols >= 2:
            danger_score += 12
            signals.append(f"💵 {currency_symbols} symboles monétaires")
        
        # 10. DÉTECTION DE MOTS RÉPÉTÉS (technique de spam)
        words = text_lower.split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Ignore les petits mots
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated = [word for word, count in word_counts.items() if count >= 3]
        if repeated:
            danger_score += len(repeated) * 5
            signals.append(f"🔁 Mots répétés : {', '.join(repeated[:3])}")
        
        # Limiter le score à 100
        danger_score = min(danger_score, 100)
        
        return danger_score, signals

# Initialisation de l'analyseur
pattern_analyzer = DangerPatternAnalyzer()

# ============================================
# CHARGEMENT DU DATASET
# ============================================

def load_dataset():
    """Charge le dataset SMS Spam Collection"""
    print("📥 Téléchargement du dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            with zip_file.open('SMSSpamCollection') as f:
                df = pd.read_csv(f, sep='\t', names=['label', 'message'], encoding='latin-1')
        
        print(f"✅ Dataset chargé : {len(df)} messages")
        print(f"   - Ham : {len(df[df['label']=='ham'])}")
        print(f"   - Spam : {len(df[df['label']=='spam'])}")
        
        return df
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        # Dataset minimal de secours
        return pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
            'message': [
                'Hello, how are you doing today?',
                'WINNER! FREE cash prize! Click NOW!!!',
                'Meeting at 3pm tomorrow, see you there',
                'Congratulations! Call +1-555-0100 to claim your prize',
                'Thanks for your help with the project',
                'URGENT: Verify your account NOW at http://fake-bank.com'
            ]
        })

# ============================================
# PRÉTRAITEMENT AVANCÉ
# ============================================

def preprocess_data(df):
    """Prépare les données pour l'entraînement"""
    print("🧹 Prétraitement des données...")
    
    # Conversion des labels
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Suppression des lignes vides
    df = df.dropna()
    
    # Suppression des doublons (améliore la qualité)
    df = df.drop_duplicates(subset=['message'])
    
    print(f"✅ {len(df)} messages uniques prêts")
    return df

# ============================================
# VECTORISATION TF-IDF AMÉLIORÉE
# ============================================

def create_features(X_train, X_test):
    """
    Vectorisation TF-IDF optimisée
    - N-grams (1,3) : mots seuls + paires + triplets
    - Plus de features : 5000 au lieu de 3000
    - Min_df : ignore les mots trop rares
    """
    print("🔢 Vectorisation TF-IDF améliorée...")
    
    global vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=5000,        # Plus de mots = plus précis
        ngram_range=(1, 3),       # Unigrams, bigrams, trigrams
        min_df=2,                 # Ignore les mots présents dans < 2 docs
        max_df=0.8,               # Ignore les mots trop fréquents
        strip_accents='unicode',  # Gère les accents
        token_pattern=r'\b\w+\b'  # Garde les mots complets
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ {X_train_vec.shape[1]} features créées")
    return X_train_vec, X_test_vec

# ============================================
# ENTRAÎNEMENT - LOGISTIC REGRESSION
# ============================================
# Pourquoi Logistic Regression au lieu de Naïve Bayes ?
# - Plus précis (92-95% vs 85-90%)
# - Gère mieux les relations entre mots
# - Toujours très rapide
# - Donne de vraies probabilités calibrées

def train_model(X_train, y_train, X_test, y_test):
    """Entraîne le modèle Logistic Regression"""
    print("🤖 Entraînement Logistic Regression...")
    
    global model, model_stats
    
    # Création du modèle
    model = LogisticRegression(
        C=1.0,              # Régularisation (1.0 = équilibré)
        max_iter=1000,      # Itérations max
        solver='lbfgs',     # Algorithme d'optimisation
        random_state=42     # Reproductibilité
    )
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    
    model_stats = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'model_type': 'Logistic Regression (Hybride)'
    }
    
    print(f"✅ Modèle entraîné !")
    print(f"   Précision : {accuracy*100:.2f}%")
    print(f"\n📊 Rapport :\n{class_report}")
    
    return model

# ============================================
# SYSTÈME DE PRÉDICTION HYBRIDE
# ============================================

def predict_hybrid(text):
    """
    Prédiction hybride intelligente
    
    Niveau 1 : ML rapide (Logistic Regression)
    Niveau 2 : Analyse de patterns si confiance < 90%
    
    Retourne : prediction, confidence, method, signals
    """
    
    # NIVEAU 1 : Prédiction ML
    text_vec = vectorizer.transform([text])
    prediction_num = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0]
    ml_confidence = max(proba) * 100
    
    # NIVEAU 2 : Analyse de patterns
    danger_score, signals = pattern_analyzer.analyze(text)
    
    # DÉCISION HYBRIDE
    # Si ML est très confiant (>90%), on lui fait confiance
    if ml_confidence >= 90:
        final_prediction = "spam" if prediction_num == 1 else "ham"
        final_confidence = ml_confidence
        method = "ml_fast"
    
    # Sinon, on combine ML + patterns
    else:
        # Pondération : 70% ML + 30% patterns
        ml_spam_score = proba[1] * 100  # Probabilité de spam selon ML
        combined_score = (ml_spam_score * 0.7) + (danger_score * 0.3)
        
        final_prediction = "spam" if combined_score >= 50 else "ham"
        final_confidence = combined_score if combined_score >= 50 else (100 - combined_score)
        method = "ml_deep_analysis"
        
        # Ajout d'un signal pour expliquer la décision
        if method == "ml_deep_analysis":
            signals.insert(0, f"🧠 Analyse approfondie (ML: {ml_spam_score:.1f}% + Patterns: {danger_score}%)")
    
    return final_prediction, final_confidence, method, signals

# ============================================
# SAUVEGARDE / CHARGEMENT
# ============================================

def save_model():
    """Sauvegarde le modèle"""
    print("💾 Sauvegarde du modèle...")
    with open('spam_model_hybrid.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer_hybrid.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✅ Modèle sauvegardé")

def load_model():
    """Charge le modèle sauvegardé"""
    global model, vectorizer
    try:
        with open('spam_model_hybrid.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer_hybrid.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Modèle chargé")
        return True
    except FileNotFoundError:
        print("⚠️ Aucun modèle sauvegardé")
        return False

# ============================================
# INITIALISATION AU DÉMARRAGE
# ============================================

@app.on_event("startup")
async def startup_event():
    """Entraîne ou charge le modèle au démarrage"""
    print("\n" + "="*60)
    print("🚀 DÉTECTEUR DE SPAM HYBRIDE v2.0")
    print("="*60 + "\n")
    
    if not load_model():
        df = load_dataset()
        df = preprocess_data(df)
        
        X = df['message']
        y = df['label_num']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}\n")
        
        X_train_vec, X_test_vec = create_features(X_train, X_test)
        train_model(X_train_vec, y_train, X_test_vec, y_test)
        save_model()
    
    print("\n" + "="*60)
    print("✅ SYSTÈME PRÊT - Hybride ML + Analyse de Patterns")
    print("="*60 + "\n")

# ============================================
# ROUTES DE L'API
# ============================================

@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "name": "Détecteur de Spam Hybride",
        "version": "2.0.0",
        "description": "IA combinant ML et analyse de patterns",
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
        "model_type": "Logistic Regression + Pattern Analysis"
    }

@app.get("/stats")
async def get_stats():
    """Retourne les statistiques"""
    if not model_stats:
        raise HTTPException(status_code=503, detail="Modèle non entraîné")
    
    return {
        "accuracy": f"{model_stats['accuracy']*100:.2f}%",
        "model_type": model_stats.get('model_type', 'Unknown'),
        "confusion_matrix": model_stats['confusion_matrix']
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(message: Message):
    """
    Endpoint principal : prédiction hybride
    
    Combine :
    - Machine Learning (Logistic Regression)
    - Analyse de patterns dangereux
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        # Prédiction hybride
        prediction, confidence, method, signals = predict_hybrid(message.text)
        
        return PredictionResponse(
            text=message.text,
            prediction=prediction,
            confidence=round(confidence, 2),
            method=method,
            danger_signals=signals
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ============================================
# LANCEMENT
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)