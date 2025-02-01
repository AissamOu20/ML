import tkinter as tk
from tkinter import messagebox, Frame
import joblib
import re
import threading

class PhishingDetector:
    def __init__(self,root):
        self.additional_results_frame = None
        self.main_frame = None
        self.text_input = None
        self.root = root
        self.root.title("Analyseur d'Email Phishing")
        self.root.geometry("800x700")
        self.root.configure(bg="#1A1A1A")

        # Initialisation de l'écran de chargement
        self.loading_label = tk.Label(root, text="Chargement du modèle ML...",
                                      font=("Consolas", 16), bg="#1A1A1A", fg="#FF0000")
        self.loading_label.pack(expand=True)

        # Initialisation des variables
        self.model = None
        self.tfidf_vectorizer = None
        self.model_status_indicator = None

        # Chargement du modèle ML en arrière-plan
        threading.Thread(target=self.load_ml_model, daemon=True).start()

    def load_ml_model(self):
        import time
        time.sleep(3)
        self.model = joblib.load('phishing_detection_model.pkl')
        self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

        self.setup_main_interface()

    def setup_main_interface(self):
        # Supprimer l'étiquette de chargement
        self.loading_label.destroy()

        # Conteneur principal
        self.main_frame = Frame(self.root, bg="#1A1A1A")
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Titre
        title = tk.Label(self.main_frame, text="ANALYSEUR DE PHISHING",
                         font=("Consolas", 24, "bold"), bg="#1A1A1A", fg="#FF0000")
        title.pack(pady=(0, 20))

        # Section d'entrée
        input_frame = Frame(self.main_frame, bg="#2D2D2D")
        input_frame.pack(fill="x", pady=(0, 20))

        input_label = tk.Label(input_frame, text="CONTENU DU MESSAGE",
                                font=("Consolas", 11), bg="#2D2D2D", fg="#888888")
        input_label.pack(pady=(15, 5), padx=15, anchor="w")

        self.text_input = tk.Text(input_frame, height=8,
                                   font=("Consolas", 12), bg="#333333", fg="#FFFFFF",
                                   insertbackground="#FF0000", bd=0)
        self.text_input.pack(fill="x", expand=True, padx=15, pady=(0, 15))

        # Bouton d'analyse
        self.analyze_button = tk.Button(input_frame, text="ANALYSER",
                                        command=self.validate_and_analyze,
                                        font=("Consolas", 12, "bold"),
                                        bg="#008000", fg="#008000",
                                        activebackground="#CC0000",
                                        activeforeground="#FFFFFF",
                                        relief="flat",
                                        padx=30, pady=8)
        self.analyze_button.pack(pady=(0, 15))

        # Section des résultats
        self.results_frame = Frame(self.main_frame, bg="#1A1A1A")
        self.results_frame.pack(fill="both", expand=True)

        # Message de détection de phishing
        self.phishing_message_label = tk.Label(self.results_frame, text="En attente de l'analyse...",
                                                font=("Consolas", 12), bg="#1A1A1A", fg="#FFFFFF")
        self.phishing_message_label.pack(anchor="s", pady=(10, 0))
        self.phishing_message_label.pack(anchor="s", pady=(10, 0))

        # Carte du score de confiance
        self.confidence_value = self.create_card(self.results_frame,
                                                 "SCORE DE CONFIANCE", "--")

        # Section du statut du modèle
        self.create_model_status_section()

        # Carte des modèles détectés avec un style personnalisé pour plusieurs lignes
        patterns_card = Frame(self.results_frame, bg="#2D2D2D", padx=15, pady=15)
        patterns_card.pack(fill="x", padx=10, pady=5)

        patterns_title = tk.Label(patterns_card, text="MODÈLES DÉTECTÉS",
                                  font=("Consolas", 11), bg="#2D2D2D", fg="#888888")
        patterns_title.pack(anchor="w")

        self.patterns_value = tk.Label(patterns_card, text="En attente de l'analyse...",
                                       font=("Consolas", 12), bg="#2D2D2D", fg="#FFFFFF",
                                       justify="left", wraplength=700)
        self.patterns_value.pack(anchor="w", pady=(10, 0))

        # Section des résultats d'analyse supplémentaires
        self.additional_results_frame = Frame(self.results_frame, bg="#2D2D2D", padx=15, pady=15)
        self.additional_results_frame.pack(fill="x", padx=10, pady=5)

        additional_title = tk.Label(self.additional_results_frame, text="RÉSULTATS D'ANALYSE ADDITIONNELS",
                                     font=("Consolas", 11), bg="#2D2D2D", fg="#888888")
        additional_title.pack(anchor="w")

        self.additional_results_value = tk.Text(self.additional_results_frame, height=6, width=60,
                                                 font=("Consolas", 12), bg="#333333", fg="#FFFFFF",
                                                 wrap="word", bd=0)
        self.additional_results_value.pack(anchor="w", pady=(10, 0))
        self.additional_results_value.config(state=tk.DISABLED)  # Désactivé au départ

    def create_model_status_section(self):
        """Créer une section compacte pour le statut du modèle de machine learning."""
        status_frame = Frame(self.results_frame, bg="#2D2D2D", padx=15, pady=15)
        status_frame.pack(fill="x", padx=10, pady=5, anchor="w")  # Aligné à gauche

        # Frame pour l'indicateur lumineux et le titre
        title_frame = Frame(status_frame, bg="#2D2D2D")
        title_frame.pack(fill="x", anchor="w")

        # Indicateur lumineux
        self.model_status_indicator = tk.Canvas(title_frame, width=20, height=20, bg="#2D2D2D", highlightthickness=0)
        self.model_status_indicator.pack(side="left", padx=(0, 10))
        self.update_model_status_indicator(True)

        # Titre
        title_label = tk.Label(title_frame, text="STATUT DU MODÈLE", font=("Consolas", 11), bg="#2D2D2D", fg="#888888")
        title_label.pack(side="left", anchor="w")

        # Statut du modèle
        self.model_status_value = tk.Label(status_frame, text="Modèle chargé avec succès",
                                           font=("Consolas", 12), bg="#2D2D2D", fg="#FFFFFF")
        self.model_status_value.pack(anchor="w", pady=(5, 0))

        # Détails supplémentaires avec une mise en forme plus précise
        model_info = (
            ("Type de modèle:", "Random Forest"),
            ("Type de vectoriseur:", "TF-IDF"),
            ("Précision de l'entraînement:", "95%")
        )

        # Créer une frame pour aligner correctement les informations
        details_frame = Frame(status_frame, bg="#2D2D2D")
        details_frame.pack(anchor="w", pady=(5, 0))

        for label, value in model_info:
            row = Frame(details_frame, bg="#2D2D2D")
            row.pack(fill="x")

            label_widget = tk.Label(row, text=label, font=("Consolas", 11), bg="#2D2D2D", fg="#FFFFFF")
            label_widget.pack(side="left", anchor="w", padx=5)

            value_widget = tk.Label(row, text=value, font=("Consolas", 11), bg="#2D2D2D", fg="#FFFFFF")
            value_widget.pack(side="left", anchor="w", padx=5)

    def update_model_status_indicator(self, is_running):
        """Met à jour l'indicateur lumineux du statut du modèle."""
        color = "green" if is_running else "red"
        self.model_status_indicator.create_oval(2, 2, 13, 13, fill=color, outline=color)

    def validate_and_analyze(self):
        text = self.text_input.get("1.0", tk.END).strip()

        if not text:
            messagebox.showerror("Erreur", "Veuillez entrer un texte à analyser.")
            return

        if not self.is_valid_input(text):
            messagebox.showerror("Erreur", "Veuillez entrer un contenu valide.\n\n" +
                                 "Le contenu valide doit inclure :\n" +
                                 "- Messages d'email\n" +
                                 "- Liens URL\n" +
                                 "- Contenu texte\n\n" +
                                 "Évitez les caractères spéciaux ou les extraits de code.")
            return

        self.analyze_text()

    def is_valid_input(self, text):
        if len(text) < 10:
            return False

        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        has_email = bool(re.search(email_pattern, text))
        has_url = bool(re.search(url_pattern, text))
        is_text = bool(re.search(r'[A-Za-z\s]{10,}', text))

        has_script = '<script' in text.lower() or 'javascript:' in text.lower()
        has_sql = 'select' in text.lower() and 'from' in text.lower()

        return (has_email or has_url or is_text) and not (has_script or has_sql)

    def analyze_text(self):
        if not self.model or not self.tfidf_vectorizer:
            messagebox.showerror("Erreur", "Le modèle n'est pas encore chargé.")
            return

        # Récupérer le texte de l'interface utilisateur
        text = self.text_input.get("1.0", tk.END).strip()

        # Transformer le texte en fonction du modèle TF-IDF
        text_vectorized = self.tfidf_vectorizer.transform([text])

        # Effectuer la prédiction avec le modèle
        prediction_proba = self.model.predict_proba(text_vectorized)

        # La classe prédite (0 pour "Légitime", 1 pour "Phishing")
        predicted_class = self.model.predict(text_vectorized)[0]

        # Probabilités pour chaque classe
        legit_prob = prediction_proba[0][0]  # Probabilité de légitimité
        phishing_prob = prediction_proba[0][1]  # Probabilité de phishing

        # Mettre à jour l'interface utilisateur avec les résultats
        prediction = 'Phishing' if predicted_class == 1 else 'Légitime'
        probabilities = [legit_prob, phishing_prob]

        self.update_phishing_message(prediction, probabilities)
    #Message apres l'analyse
    def update_phishing_message(self, prediction, probabilities):
        if prediction == 'Phishing':
            self.phishing_message_label.config(text="Phishing Détecté!", fg="#FF0000")
            self.confidence_value.config(text=f"Score de Confiance : {probabilities[1] * 100:.2f}%")
            self.patterns_value.config(text="Modèles détectés : Liens suspects, Expéditeur inhabituel")
            additional_info = (
                "Résultat de l'analyse : Email potentiel de phishing.\n\n"
                "Recommandations :\n"
                "- Ne cliquez sur aucun lien ni ne téléchargez de pièces jointes.\n"
                "- Vérifiez l'expéditeur et soyez prudent avec les emails inconnus."
            )
            self.update_additional_results(additional_info)
        else:
            self.phishing_message_label.config(text="Email Légitime", fg="#00FF00")
            self.confidence_value.config(text=f"Score de Confiance : {probabilities[0] * 100:.2f}%")
            self.patterns_value.config(text="Modèles détectés : Aucun lien suspect.")
            additional_info = (
                "Résultat de l'analyse : Email vérifié comme légitime.\n\n"
                "Recommandations :\n"
                "- Continuez à suivre les bonnes pratiques de sécurité."
            )
            self.update_additional_results(additional_info)

    def update_additional_results(self, additional_info):
        self.additional_results_value.config(state=tk.NORMAL)
        self.additional_results_value.delete("1.0", tk.END)
        self.additional_results_value.insert("1.0", additional_info)
        self.additional_results_value.config(state=tk.DISABLED)

    @staticmethod
    def create_card(parent_frame, title, value):
        card = Frame(parent_frame, bg="#2D2D2D", padx=15, pady=15)
        card.pack(fill="x", padx=10, pady=5)

        card_title = tk.Label(card, text=title, font=("Consolas", 11), bg="#2D2D2D", fg="#888888")
        card_title.pack(anchor="w")

        card_value = tk.Label(card, text=value, font=("Consolas", 12), bg="#2D2D2D", fg="#FFFFFF")
        card_value.pack(anchor="w", pady=(10, 0))

        return card_value

# Lancer l'application
root = tk.Tk()
app = PhishingDetector(root)
root.mainloop()
