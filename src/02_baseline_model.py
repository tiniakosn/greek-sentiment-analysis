import pandas as pd
import re # Βιβλιοθήκη για Regular Expressions (για να βρίσκουμε μοτίβα στο κείμενο, π.χ. σημεία στίξης)
from sklearn.model_selection import train_test_split # Για να κόψουμε τα δεδομένα σε train/test
from sklearn.feature_extraction.text import TfidfVectorizer # Για να κάνουμε τις λέξεις αριθμούς
from sklearn.linear_model import LogisticRegression # Το Baseline μοντέλο μας
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Οι μετρικές μας

# ---------------------------------------------------------
# 1. Συνάρτηση Καθαρισμού Κειμένου
# ---------------------------------------------------------
def clean_text(text):
    """
    Καθαρίζει το ελληνικό κείμενο κάνοντάς το πεζό και αφαιρώντας σημεία στίξης.
    """
    text = text.lower() # Μετατροπή σε πεζά 
    # Το re.sub αντικαθιστά οτιδήποτε ΔΕΝ  είναι γράμμα ή αριθμός (\\w) ή κενό (\\s) με κενό string ('')
    text = re.sub(r'[^\w\s]', '', text) 
    return text

# ---------------------------------------------------------
# 2. Κύρια ροή εκπαίδευσης
# ---------------------------------------------------------
def run_baseline():
    print("1. Φόρτωση δεδομένων (Αθηνόραμα)...")
    # Διαβάζουμε το ΜΕΓΑΛΟ CSV
    # Προσθέτουμε το encoding='utf-8' για να διαβάσει σωστά τα ελληνικά
    try:
        df = pd.read_csv("data/Athinorama_50.000.csv", encoding='utf-8')
    except UnicodeDecodeError:
        # Αν το utf-8 αποτύχει, σημαίνει ότι το αρχείο σώθηκε με κωδικοποίηση Windows
        print("Το UTF-8 απέτυχε, δοκιμή με Windows-1253...")
        df = pd.read_csv("data/Athinorama_50.000.csv", encoding='windows-1253')
    
    # ---  Καθαρισμός ετικετών ---
    # Κάνουμε όλα τα γράμματα πεζά (lower) και κόβουμε τα κενά αριστερά-δεξιά (strip)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    
    # Τώρα το mapping θα πετύχει σίγουρα
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})
    
    # Διαγράφουμε τυχόν πραγματικά κενές γραμμές
    df = df.dropna()
    
    print(f"Διαβάστηκαν επιτυχώς {len(df)} κριτικές!")
    
    print("2. Καθαρισμός κειμένων (Υπομονή 1-2 λεπτά!)...")
    # Εφαρμόζουμε τον καθαρισμό στη στήλη 'review'
    df['clean_text'] = df['review'].apply(clean_text)
    
    # Ορίζουμε τα X και y
    X = df['clean_text']
    y = df['label']
    
    print("3. Διαχωρισμός σε Train (80%) και Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("4. Διανυσματοποίηση με TF-IDF (Λέξεις -> Αριθμοί)...")
    vectorizer = TfidfVectorizer(max_features=10000) 
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("5. Εκπαίδευση Μοντέλου (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000) 
    model.fit(X_train_vec, y_train)
    
    print("6. Αξιολόγηση Μοντέλου...")
    predictions = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ BASELINE ---")
    print(f"Accuracy (Ακρίβεια): {acc * 100:.2f}%")
    print("\nΑναλυτικό Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_baseline()