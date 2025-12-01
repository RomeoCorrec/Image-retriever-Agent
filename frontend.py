import streamlit as st
import requests
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IA Recherche Photos", 
    page_icon="📸", 
    layout="wide"
)

# URL de ton API (Backend)
API_URL = "http://127.0.0.1:8000"

st.title("📸 Agent de Recherche d'Images")
st.markdown("---")

# Dans frontend.py

# --- BARRE LATÉRALE : GESTION ---
with st.sidebar:
    st.header("⚙️ Gestion de la Galerie")
    
    tab1, tab2 = st.tabs(["Ajout Photo", "Nouveau Visage"])
    
    # --- ONGLET 1 : Ajouter une image (Drag & Drop) ---
    with tab1:
        st.caption("Glissez une image ici pour l'ajouter.")
        
        # WIDGET DRAG & DROP
        uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'], key="uploader_img")
        
        if uploaded_file is not None:
            if st.button("Indexser l'image", type="primary"):
                with st.spinner("Envoi et analyse..."):
                    try:
                        # Préparation de l'envoi du fichier (Multipart)
                        # On met "application/octet-stream" par défaut si le type est None
                        mime_type = uploaded_file.type if uploaded_file.type else "application/octet-stream"
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
                        
                        response = requests.post(f"{API_URL}/add_image", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"✅ Image '{data.get('filename')}' ajoutée !")
                            people = data.get("detected_people", [])
                            if people:
                                st.info(f"👤 Reconnus : {', '.join(people)}")
                            else:
                                st.caption("Aucun visage connu détecté.")
                        else:
                            st.error(f"Erreur API : {response.text}")
                    except Exception as e:
                        st.error(f"Erreur connexion : {e}")

    # --- ONGLET 2 : Apprendre un visage (Drag & Drop) ---
    with tab2:
        st.caption("Apprendre un visage.")
        
        face_file = st.file_uploader("Photo portrait", type=['jpg', 'jpeg', 'png'], key="uploader_face")
        name_input = st.text_input("Nom de la personne")
        
        if st.button("Mémoriser", disabled=(not face_file or not name_input)):
            with st.spinner("Apprentissage..."):
                try:
                    # Envoi fichier + nom (Form data)
                    files = {"file": (face_file.name, face_file.getvalue(), face_file.type)}
                    data = {"person_name": name_input}
                    
                    response = requests.post(f"{API_URL}/add_face", files=files, data=data)
                    
                    if response.status_code == 200:
                        st.success(f"🎉 Visage de {name_input} mémorisé !")
                    else:
                        st.error(f"Erreur : {response.text}")
                except Exception as e:
                    st.error(f"Erreur connexion : {e}")

# --- ZONE PRINCIPALE : CHAT ---

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je peux retrouver vos photos. Demandez-moi par exemple : 'Une photo de Romeo à la montagne'."}]

# Affichage des messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Si le message contient une image (stockée dans une clé spéciale 'image'), on l'affiche
        if "image" in msg:
            st.image(msg["image"], width=500)

# Zone de saisie
if prompt := st.chat_input("Décrivez la photo que vous cherchez..."):
    # 1. Afficher la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Appeler l'API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Recherche dans vos souvenirs..."):
            try:
                api_res = requests.post(f"{API_URL}/ask_agent", json={"query": prompt})
                
                if api_res.status_code == 200:
                    response_text = api_res.json().get("response", "Pas de réponse.")
                    
                    # Logique simple pour trouver et afficher l'image
                    found_image_path = None
                    words = response_text.replace("'", "").replace('"', '').split()
                    for word in words:
                        # On cherche grossièrement si un mot ressemble à un chemin d'image valide
                        if word.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and (os.path.exists(word) or "/" in word or "\\" in word):
                            found_image_path = word
                            break
                    
                    message_placeholder.markdown(response_text)
                    
                    # Sauvegarde dans l'historique
                    msg_data = {"role": "assistant", "content": response_text}
                    
                    if found_image_path:
                        # Si on a trouvé un chemin, on essaie de l'afficher
                        if os.path.exists(found_image_path):
                            st.image(found_image_path, caption="Image trouvée", width=500)
                            msg_data["image"] = found_image_path # On sauvegarde l'image dans l'historique aussi
                        else:
                            st.warning(f"J'ai trouvé ce chemin, mais je n'arrive pas à lire le fichier : {found_image_path}")
                    
                    st.session_state.messages.append(msg_data)

                else:
                    err_msg = f"Erreur de l'agent ({api_res.status_code})"
                    message_placeholder.error(err_msg)
            
            except Exception as e:
                message_placeholder.error(f"Erreur de connexion : {e}")