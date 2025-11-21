from firebase_admin import firestore

# Get Firestore client
db = firestore.client()

# --------- Add User Data ----------
def set_user_profile(uid, data):
    """Add or update user profile data"""
    db.collection("users").document(uid).set(data)
    return True

# --------- Get User Data ----------
def get_user_profile(uid):
    """Fetch user profile"""
    doc = db.collection("users").document(uid).get()
    return doc.to_dict() if doc.exists else None

# --------- Add Stock Watchlist ----------
def add_to_watchlist(uid, stock_symbol):
    """Append a stock symbol into user's watchlist array"""
    db.collection("users").document(uid).update({
        "watchlist": firestore.ArrayUnion([stock_symbol])
    })

# --------- Remove From Watchlist ----------
def remove_from_watchlist(uid, stock_symbol):
    """Remove stock from watchlist"""
    db.collection("users").document(uid).update({
        "watchlist": firestore.ArrayRemove([stock_symbol])
    })

# --------- Store Transaction History ----------
def add_transaction(uid, transaction_data):
    """Store buy/sell transaction record"""
    db.collection("users").document(uid).collection("transactions").add(transaction_data)
