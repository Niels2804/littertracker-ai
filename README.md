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

---

## 🐳 Wil je het via Docker runnen?
1. **Maak een docker image aan**
   ```docker build -t littertracker-ai .```
2. **Start een Docker container**
   ```docker run -p 8000:8000 littertracker-ai```
   Hiermee wordt de app beschikbaar via http://localhost:8000
3. **Stop een Docker container?**
   ```docker ps```
   of
   ```docker stop littertracker-ai``` 
   of 
   ```docker stop <container-id>```

Je kunt de container ook lokaal draaien via Docker Desktop. Zorg er dan voor dat poort 8000 via de container-port settings wordt doorgestuurd (exposed) naar je lokale machine, zodat je de API kunt bereiken via ```localhost:8000```.