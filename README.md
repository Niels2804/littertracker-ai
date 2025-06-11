# 🧠 TrashTracker AI – Predictie API

Dit project gebruikt **FastAPI** en **machine learning** om voorspellingen te doen op basis van JSON-data. Het is ontworpen om lokaal of in een Docker-container te draaien.

---

## 📁 Inhoud

- [Installatie (lokaal)](#-installatie-lokaal)
- [Starten van de API](#-api-starten)
- [API endpoints](#-api-endpoints)
- [Docker gebruiken](#-docker-gebruiken)
- [Projectstructuur](#-projectstructuur)
- [Toekomstige uitbreidingen](#-toekomstige-uitbreidingen)

---

## ⚙️ Installatie (lokaal)

> Zorg dat je Python 3.12.5+ geïnstalleerd hebt.

1. **Navigeer naar de map ```trashtracker-ai```**  
   ```cd trashtracker-ai```
2. **Installeer alle requirements door de volgende commando in de terminal uit te voeren**
   ```pip install -r requirements.txt```
3. **Voer de volgende commando in de terminal uit om de FastAPI uit te voeren**
   ```uvicorn main:app --reload```
    - main:app = main.py bevat het FastAPI app object
    - --reload zorgt dat de server automatisch herstart bij codewijzigingen