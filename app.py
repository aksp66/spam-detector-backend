# ============================================
# APP.PY - DÉTECTEUR DE SPAM v3.0 PRODUCTION-READY
# ============================================
# Nouveautés :
# - Dataset moderne multilingue (FR + EN)
# - Système de whitelist
# - Gestion de l'incertitude
# - Feedback pour amélioration continue
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import requests
import io
from urllib.parse import urlparse
from typing import Optional

# ============================================
# CONFIGURATION
# ============================================

app = FastAPI(
    title="Détecteur de Spam v3.0",
    description="IA moderne avec whitelist et gestion d'incertitude",
    version="3.0.0"
)

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
    user_domain: Optional[str] = None  # Domaine de confiance de l'utilisateur

class PredictionResponse(BaseModel):
    text: str
    prediction: str  # "spam", "ham", "uncertain"
    confidence: float
    certainty_level: str  # "high", "medium", "low"
    method: str
    danger_signals: list
    recommendation: str

class FeedbackRequest(BaseModel):
    text: str
    predicted: str
    actual: str  # Ce que l'utilisateur dit que c'était vraiment

# ============================================
# VARIABLES GLOBALES
# ============================================

model = None
vectorizer = None
model_stats = {}
user_feedback = []  # Stocke les retours utilisateurs

# ============================================
# WHITELIST - Domaines de confiance
# ============================================

TRUSTED_DOMAINS = [
    # Domaines d'éducation
    'esgis.edu', 'esgis.tg', 'universite.edu',
    
    # Domaines gouvernementaux
    'gouv.tg', 'gouv.fr', 'gov',
    
    # Grandes institutions
    'google.com', 'microsoft.com', 'apple.com',
    'facebook.com', 'linkedin.com',
    
    # Banques reconnues
    'bnpparibas.com', 'creditagricole.fr',
    
    # Services légitimes
    'paypal.com', 'stripe.com', 'amazon.com'
]

def is_trusted_domain(url):
    """
    Vérifie si une URL provient d'un domaine de confiance
    """
    try:
        domain = urlparse(url).netloc.lower()
        # Enlève www.
        domain = domain.replace('www.', '')
        
        # Vérifie si le domaine ou son parent est dans la whitelist
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith('.' + trusted):
                return True
        return False
    except:
        return False

# ============================================
# DATASET MODERNE MULTILINGUE
# ============================================

def create_modern_dataset():
    """
    Crée un dataset moderne avec spams français et anglais
    Inclut : crypto, phishing, COVID, arnaques sociales
    """
    print("📦 Création du dataset moderne...")
    
    # SPAMS MODERNES (2020-2024)
    spam_messages = [
        # Crypto & Finance (FR)
        "🚀 BITCOIN GRATUIT! Investissez maintenant et gagnez 10000€ en 24h! Cliquez: bit.ly/crypto-WIN",
        "Félicitations! Votre compte Binance a gagné 5 ETH. Réclamez maintenant sur: tiny.cc/binance",
        "ALERTE: Votre compte bancaire sera bloqué. Vérifiez vos informations: secure-bank-tg.com",
        "💰 Gagnez 5000€/mois en travaillant 2h par jour! Formation gratuite: bit.ly/rich-now",
        "URGENT: PayPal Security Alert. Confirmez votre identité: paypa1-secure.com",
        
        # Phishing (FR + EN)
        "Votre colis DHL est en attente. Payez 2€ de frais: dhl-delivery-tg.com",
        "Amazon: Votre commande #12345 a un problème de paiement. Mettez à jour: amaz0n-pay.com",
        "Netflix: Votre abonnement expire. Renouvelez maintenant: netflix-renouvellement.com",
        "Orange: Facture impayée de 89€. Réglez avant coupure: orange-facture-tg.com",
        "WhatsApp: Votre compte sera désactivé. Vérifiez ici: whatsapp-verify.net",
        
        # Arnaques sociales (FR)
        "Maman j'ai perdu mon téléphone, appelle moi sur ce numéro: +33612345678",
        "Bonjour, je suis votre nouvelle collègue Emma. On peut se rencontrer? Voici ma photo: bit.ly/emma-photo",
        "URGENT: Votre fils a eu un accident. Envoyez 500€ immédiatement à ce numéro.",
        "Félicitations! Vous avez gagné un iPhone 15 Pro. Réclamez-le: gagnant-iphone.com",
        "🎁 Carrefour vous offre un bon de 100€! Participez: carrefour-promo-tg.com",
        
        # COVID & Santé
        "VACCIN COVID disponible à domicile. Réservez: vaccination-express.com",
        "Test COVID GRATUIT. Commandez en ligne: test-covid-gratuit.net",
        "Masques FFP2 certifiés à 0.50€. Stock limité: masques-promo.com",
        
        # Spam commercial agressif
        "SOLDES -90%!!! Tout doit disparaître! CLIQUEZ MAINTENANT: mega-soldes.com",
        "Pilule miracle: Perdez 10kg en 1 semaine GARANTI! bit.ly/mincir",
        "Agrandissez votre pénis naturellement! Résultats en 7 jours: bit.ly/natural",
        
        # Arnaques emploi
        "Travail à domicile: 3000€/mois FACILE. Aucune expérience requise: job-facile.net",
        "Recrutement URGENT: 10 postes disponibles. Inscrivez-vous: recrutement-express.com",
        
        # Spams avec émojis excessifs
        "🎉🎊🎁💰💵 GAGNEZ GROS!!! 🚀🚀🚀 MAINTENANT!!! ⚡⚡⚡ bit.ly/WIN",
        "😍😍😍 INCROYABLE!!! 💎💎💎 Cliquez vite!!! ⏰⏰⏰",
        
        # Phishing sophistiqué
        "Bonjour, suite à notre conversation téléphonique, voici le lien du document: dropb0x.com/doc123",
        "Votre banque: Transaction suspecte détectée. Vérifiez: secure.banque-tg.com",
        
        # Anglais modernes
        "WINNER! You won $10,000! Claim now: bit.ly/cash-winner",
        "Your Amazon package is delayed. Track here: amzn-track.com",
        "Bitcoin investment: Turn $100 into $10,000 overnight! invest-btc.net",
        "URGENT: Your Apple ID is locked. Unlock: apple-security.net",
        "FREE iPhone 15! Limited stock. Order now: free-iphone.com",
        "Work from home: Earn $5000/month EASY! job-remote.com"
    ]
    
    # MESSAGES LÉGITIMES (HAM)
    ham_messages = [
        # Messages d'école (FR)
        "Bonjour, le cours de demain est déplacé en salle B203. Cordialement, Prof Martin",
        "Rappel: Remise des projets vendredi à 17h. Bonne chance à tous!",
        "Les résultats des examens seront disponibles lundi sur l'ENT.",
        "N'oubliez pas la réunion de rentrée jeudi à 14h à l'amphithéâtre.",
        
        # Messages personnels (FR)
        "Salut! Tu viens au ciné ce soir? On se retrouve à 20h devant le cinéma.",
        "Merci pour ton aide hier, ça m'a vraiment aidé pour le projet!",
        "Papa, je rentre vers 18h. Bisous!",
        "Joyeux anniversaire! On se voit samedi pour fêter ça?",
        
        # Messages professionnels (FR)
        "Bonjour, voici le compte-rendu de la réunion d'hier. Cordialement,",
        "La présentation est prête. Je vous l'envoie en pièce jointe.",
        "Pouvez-vous me confirmer votre disponibilité pour lundi?",
        "Merci pour votre retour. Je prends note de vos remarques.",
        
        # Notifications légitimes (FR)
        "Votre colis sera livré demain entre 9h et 12h. Suivi: [lien officiel La Poste]",
        "Orange: Votre facture de 45.99€ est disponible sur votre espace client.",
        "Netflix: Votre paiement a été accepté. Merci de votre abonnement.",
        "Banque: Virement reçu de 500€ de Marie Dupont.",
        
        # Messages avec liens légitimes
        "Voici le lien vers le document Google Drive: docs.google.com/document/d/abc123",
        "Regarde cette vidéo intéressante: youtube.com/watch?v=abc123",
        "Article intéressant sur le ML: medium.com/article-ml-2024",
        
        # Anglais légitimes
        "Hey, are we still meeting tomorrow at 3pm?",
        "Thanks for your help with the project!",
        "The meeting has been rescheduled to Friday.",
        "Your order has been shipped. Track at: amazon.com/track/order123",
        "Don't forget to bring the documents tomorrow.",
        "See you at the conference next week!",
        
        # Conversations normales
        "LOL, c'était trop drôle hier! 😂",
        "Ok, je te rappelle dans 5 minutes.",
        "Bonne soirée! À demain 👋",
        "Bien reçu, merci!",
        "Tu as raison, c'est une bonne idée.",
        
        # Contexte étudiant
        "La cantine est fermée aujourd'hui pour travaux.",
        "Groupe de révision à la bibliothèque à 15h. Qui vient?",
        "Les inscriptions pour le voyage d'études sont ouvertes.",
        "Sortie pédagogique annulée à cause de la pluie."
    ]
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'message': spam_messages + ham_messages,
        'label': ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    })
    
    print(f"✅ Dataset créé : {len(df)} messages")
    print(f"   - Spam : {len(spam_messages)}")
    print(f"   - Ham : {len(ham_messages)}")
    
    return df

def try_load_kaggle_dataset():
    """
    Essaie de charger un dataset depuis Kaggle
    Fallback si échec
    """
    try:
        print("🔍 Tentative de téléchargement Kaggle...")
        # Note: Nécessite kaggle.json configuré
        import kaggle
        kaggle.api.authenticate()
        # Dataset populaire de spam SMS
        kaggle.api.dataset_download_files(
            'uciml/sms-spam-collection-dataset',
            path='./data',
            unzip=True
        )
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        print("✅ Dataset Kaggle chargé!")
        return df
    except Exception as e:
        print(f"⚠️ Kaggle non disponible: {e}")
        return None

def load_dataset_smart():
    """
    Stratégie intelligente de chargement
    1. Essaie Kaggle
    2. Utilise dataset moderne intégré
    """
    # Essai Kaggle
    df_kaggle = try_load_kaggle_dataset()
    
    # Dataset moderne
    df_modern = create_modern_dataset()
    
    # Combiner si Kaggle dispo
    if df_kaggle is not None:
        print("🔄 Fusion Kaggle + Dataset moderne...")
        df = pd.concat([df_kaggle, df_modern], ignore_index=True)
    else:
        df = df_modern
    
    return df

# ============================================
# ANALYSEUR DE PATTERNS (amélioré)
# ============================================

class AdvancedPatternAnalyzer:
    """
    Analyseur avancé avec détection de contexte
    """
    
    def __init__(self):
        self.urgency_words = [
            'urgent', 'now', 'immediately', 'hurry', 'quick', 'fast',
            'limited', 'expire', 'last chance', 'act now',
            'maintenant', 'vite', 'urgent', 'immédiatement', 'rapide'
        ]
        
        self.money_words = [
            'free', 'win', 'winner', 'cash', 'prize', 'earn', 'profit',
            'million', 'bitcoin', 'crypto', 'invest', 'rich',
            'gratuit', 'gagner', 'gagnant', 'argent', 'euros', 'prize'
        ]
        
        self.action_words = [
            'click', 'call', 'verify', 'confirm', 'update', 'download',
            'register', 'claim', 'redeem', 'subscribe',
            'cliquer', 'appeler', 'vérifier', 'confirmer', 'télécharger'
        ]
        
        self.phishing_words = [
            'account', 'password', 'security', 'suspended', 'locked',
            'verify', 'confirm', 'alert', 'warning',
            'compte', 'mot de passe', 'sécurité', 'bloqué', 'alerte'
        ]
    
    def analyze(self, text):
        """Analyse complète avec scoring intelligent"""
        text_lower = text.lower()
        danger_score = 0
        signals = []
        
        # 1. URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        shortened = re.findall(r'\b(?:bit\.ly|tinyurl|goo\.gl|ow\.ly|t\.co|tiny\.cc)/\w+', text_lower)
        
        # Vérifier si URLs sont de confiance
        trusted_urls = sum(1 for url in urls if is_trusted_domain(url))
        suspicious_urls = len(urls) - trusted_urls + len(shortened)
        
        if suspicious_urls > 0:
            danger_score += suspicious_urls * 25
            signals.append(f"🔗 {suspicious_urls} URL(s) suspecte(s)")
        elif trusted_urls > 0:
            danger_score -= 10  # Bonus pour domaine de confiance
            signals.append(f"✅ URL(s) de confiance détectée(s)")
        
        # 2. Numéros de téléphone
        phones = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text)
        if phones:
            danger_score += len(phones) * 15
            signals.append(f"📞 {len(phones)} numéro(s)")
        
        # 3. MAJUSCULES
        upper_ratio = len([c for c in text if c.isupper()]) / max(len(text), 1)
        if upper_ratio > 0.4:
            danger_score += 30
            signals.append("🔊 Texte majoritairement en MAJUSCULES")
        
        # 4. Exclamations
        exclamations = text.count('!')
        if exclamations >= 3:
            danger_score += min(exclamations * 5, 25)
            signals.append(f"❗ {exclamations} points d'exclamation")
        
        # 5. Émojis excessifs
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', text))
        if emoji_count >= 5:
            danger_score += 15
            signals.append(f"🎨 {emoji_count} émojis (excessif)")
        
        # 6. Mots d'urgence
        urgency = sum(1 for word in self.urgency_words if word in text_lower)
        if urgency > 0:
            danger_score += urgency * 10
            signals.append(f"⏰ {urgency} mot(s) d'urgence")
        
        # 7. Mots d'argent
        money = sum(1 for word in self.money_words if word in text_lower)
        if money > 0:
            danger_score += money * 12
            signals.append(f"💰 {money} mot(s) financier(s)")
        
        # 8. Demandes d'action
        actions = sum(1 for word in self.action_words if word in text_lower)
        if actions > 0:
            danger_score += actions * 8
            signals.append(f"👆 {actions} demande(s) d'action")
        
        # 9. Indicateurs de phishing
        phishing = sum(1 for word in self.phishing_words if word in text_lower)
        if phishing >= 2:
            danger_score += phishing * 15
            signals.append(f"🎣 {phishing} indicateur(s) de phishing")
        
        # 10. Symboles monétaires
        currency = len(re.findall(r'[$€£¥₹]', text))
        if currency >= 2:
            danger_score += 10
            signals.append(f"💵 {currency} symboles monétaires")
        
        return min(danger_score, 100), signals

pattern_analyzer = AdvancedPatternAnalyzer()

# ============================================
# MACHINE LEARNING - Random Forest
# ============================================

def train_model(X_train, y_train, X_test, y_test):
    """
    Entraîne Random Forest (meilleur que Logistic Regression)
    """
    print("🌳 Entraînement Random Forest...")
    
    global model, model_stats
    
    model = RandomForestClassifier(
        n_estimators=100,     # 100 arbres
        max_depth=20,         # Profondeur max
        min_samples_split=5,  # Min échantillons pour split
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    
    model_stats = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'model_type': 'Random Forest v3.0'
    }
    
    print(f"✅ Modèle entraîné : {accuracy*100:.2f}%")
    print(f"\n{class_report}")
    
    return model

# ============================================
# PRÉDICTION INTELLIGENTE AVEC INCERTITUDE
# ============================================

def predict_intelligent(text, user_domain=None):
    """
    Prédiction avec gestion de l'incertitude
    """
    # NIVEAU 0 : Whitelist
    urls = re.findall(r'http[s]?://[^\s]+', text)
    for url in urls:
        if is_trusted_domain(url):
            return "ham", 95.0, "high", "whitelist", ["✅ Domaine de confiance détecté"], "Message d'un domaine vérifié"
    
    # NIVEAU 1 : ML
    text_vec = vectorizer.transform([text])
    prediction_num = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0]
    ml_confidence = max(proba) * 100
    
    # NIVEAU 2 : Patterns
    danger_score, signals = pattern_analyzer.analyze(text)
    
    # DÉCISION HYBRIDE
    if ml_confidence >= 85:
        # Haute confiance
        final_pred = "spam" if prediction_num == 1 else "ham"
        final_conf = ml_confidence
        certainty = "high"
        method = "ml_confident"
        
        if final_pred == "spam":
            recommendation = "⚠️ Message très probablement dangereux. Ne cliquez pas."
        else:
            recommendation = "✅ Message probablement légitime."
    
    elif ml_confidence >= 60:
        # Confiance moyenne - combiner ML + patterns
        ml_spam_score = proba[1] * 100
        combined = (ml_spam_score * 0.65) + (danger_score * 0.35)
        
        final_pred = "spam" if combined >= 50 else "ham"
        final_conf = combined if combined >= 50 else (100 - combined)
        certainty = "medium"
        method = "ml_pattern_combined"
        
        recommendation = "⚠️ Incertitude détectée. Vérifiez la source avant d'agir."
    
    else:
        # Faible confiance - zone d'incertitude
        final_pred = "uncertain"
        final_conf = 50.0
        certainty = "low"
        method = "uncertain"
        signals.insert(0, "❓ L'IA n'est pas sûre de ce message")
        
        recommendation = "❓ INCERTAIN : Vérifiez manuellement l'expéditeur et le contenu. En cas de doute, ne cliquez pas."
    
    return final_pred, final_conf, certainty, method, signals, recommendation

# ============================================
# INITIALISATION
# ============================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🚀 DÉTECTEUR DE SPAM v3.0 - PRODUCTION")
    print("="*60 + "\n")
    
    global vectorizer, model
    
    # Charger dataset
    df = load_dataset_smart()
    
    # Prétraitement
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna().drop_duplicates(subset=['message'])
    
    # Split
    X = df['message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorisation
    print("🔢 Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85,
        strip_accents='unicode'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Entraînement
    train_model(X_train_vec, y_train, X_test_vec, y_test)
    
    print("\n" + "="*60)
    print("✅ SYSTÈME PRÊT - v3.0 PRODUCTION")
    print("="*60 + "\n")

# ============================================
# ROUTES API
# ============================================

@app.get("/")
async def root():
    return {
        "name": "Détecteur de Spam v3.0",
        "version": "3.0.0",
        "features": [
            "Dataset moderne multilingue",
            "Whitelist de domaines",
            "Gestion de l'incertitude",
            "Random Forest classifier"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "version": "3.0.0"
    }

@app.get("/stats")
async def stats():
    if not model_stats:
        raise HTTPException(status_code=503, detail="Modèle non entraîné")
    return model_stats

@app.post("/predict", response_model=PredictionResponse)
async def predict(message: Message):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        pred, conf, cert, method, signals, recommendation = predict_intelligent(
            message.text,
            message.user_domain
        )
        
        return PredictionResponse(
            text=message.text,
            prediction=pred,
            confidence=round(conf, 2),
            certainty_level=cert,
            method=method,
            danger_signals=signals,
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(fb: FeedbackRequest):
    """
    Collecte les retours utilisateurs pour amélioration
    """
    user_feedback.append({
        'text': fb.text,
        'predicted': fb.predicted,
        'actual': fb.actual
    })
    return {"message": "Merci pour votre retour!", "total_feedback": len(user_feedback)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)