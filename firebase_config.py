# firebase_config.py

import pyrebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

firebaseConfig = {
  "apiKey": "AIzaSyDWT0YnQqdFC9KA-hLDdDSpvGrk3zLlmWo",
  "authDomain": "stock-sense-project.firebaseapp.com",
  "projectId": "stock-sense-project",
  "storageBucket": "stock-sense-project.appspot.com",
  "messagingSenderId": "423833753864",
  "appId": "1:423833753864:web:a32af956e161bac98574bd",
  "measurementId": "G-YVW1MPQWLS",
  "databaseURL": "https://stock-sense-project-default-rtdb.firebaseio.com/"
}

# Firebase client-side auth
firebase = pyrebase.initialize_app(firebaseConfig)
auth_user = firebase.auth()

# Firebase admin SDK
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()
