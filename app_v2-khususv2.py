import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
from torchcrf import CRF
import steamreviews
import json
import requests
import re
import os
import time
import random
import plotly.express as px

# VENV: .\.venv\Scripts\Activate.ps1

# ==========================================
# 1. DEFINISI ARSITEKTUR MODEL
# ==========================================
class SpatialDropout1D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

class NERModelCRF(nn.Module):
    def __init__(self, model_type, embedding_matrix, num_tags, pad_idx, max_len=64):
        super(NERModelCRF, self).__init__()
        self.pad_idx = pad_idx
        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=True,
            padding_idx=pad_idx
        )
        
        hidden_size = 64
        
        if model_type == 'BiLSTM':
            self.spatial_dropout = SpatialDropout1D(0.35)
            self.rnn = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
            self.rnn_dropout = nn.Dropout(0.5)
        elif model_type == 'BiGRU':
            self.spatial_dropout = SpatialDropout1D(0.30)
            self.rnn = nn.GRU(embed_dim, hidden_size, bidirectional=True, batch_first=True)
            self.rnn_dropout = nn.Dropout(0.4)
        elif model_type == 'SimpleRNN':
            self.spatial_dropout = SpatialDropout1D(0.20)
            self.rnn = nn.RNN(embed_dim, hidden_size, bidirectional=True, batch_first=True)
            self.rnn_dropout = nn.Dropout(0.3)

        self.mha = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        
        self.extra_dense = nn.Linear(hidden_size*2, 32)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(32, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, tags=None):
        mask = (x != self.pad_idx)
        embs = self.embedding(x)
        embs = self.spatial_dropout(embs)
        
        lengths = mask.sum(dim=1).cpu()
        lengths = torch.where(lengths == 0, torch.tensor(1), lengths) 

        packed_embs = nn.utils.rnn.pack_padded_sequence(
            embs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_rnn_out, _ = self.rnn(packed_embs)
        
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_rnn_out, batch_first=True, total_length=x.size(1)
        )
        
        rnn_out = self.rnn_dropout(rnn_out)
        attn_out, _ = self.mha(rnn_out, rnn_out, rnn_out, key_padding_mask=~mask)
        out = rnn_out + attn_out
        out = self.layer_norm(out)
        out = self.extra_dense(out)
        out = self.relu(out)
        emissions = self.classifier(out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask.byte(), reduction='mean')
            return loss
        else:
            preds = self.crf.decode(emissions, mask=mask.byte())
            return preds

# ==========================================
# 2. FUNGSI UNTUK LOAD MODEL & EKSTRAKSI
# ==========================================
@st.cache_resource
def load_model(model_name):
    device = torch.device('cpu')
    
    checkpoint_path = f"models/{model_name}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    w2i = checkpoint["word2idx"]
    t2i = checkpoint["tag2idx"]
    i2t = checkpoint["idx2tag"]
    config = checkpoint["config"]
    m_type = checkpoint["model_type"]
    
    vocab_size = len(w2i)
    dummy_embedding = np.zeros((vocab_size, 300))
    pad_idx = w2i["PAD"]
    
    model = NERModelCRF(m_type, dummy_embedding, len(t2i), pad_idx, max_len=config["max_len"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, w2i, i2t, config, device

def extract_entities_robust(predictions):
    entities = []
    current_label = None
    current_phrase = []
    
    for word, tag in predictions:
        if tag == 'O':
            if current_label is not None:
                entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
                current_label = None
                current_phrase = []
                
        elif tag.startswith('B-'):
            if current_label is not None:
                entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
            current_label = tag.split('-', 1)[1]
            current_phrase = [word]
            
        elif tag.startswith('I-'):
            label = tag.split('-', 1)[1]
            if current_label == label:
                current_phrase.append(word)
            else:
                if current_label is not None:
                    entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
                current_label = label
                current_phrase = [word]
                
        elif tag.startswith('L-'):
            label = tag.split('-', 1)[1]
            if current_label == label:
                current_phrase.append(word)
                entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
            else:
                if current_label is not None:
                    entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
                entities.append({"label": label, "phrase": word})
            current_label = None
            current_phrase = []
            
        elif tag.startswith('U-'):
            if current_label is not None:
                entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
            label = tag.split('-', 1)[1]
            entities.append({"label": label, "phrase": word})
            current_label = None
            current_phrase = []
            
    if current_label is not None:
        entities.append({"label": current_label, "phrase": " ".join(current_phrase)})
        
    return entities

# ==========================================
# 3. FUNGSI PREDIKSI & DATA CLEANING
# ==========================================
# Fungsi untuk input teks tunggal (Satu per satu)
def predict_ner(text, model, w2i, i2t, max_len, device):
    text = str(text).strip()
    text = re.sub(r'([:.,!?()])', r' \1 ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    if not words:
        return []
        
    x_idx = [w2i.get(w, w2i["UNK"]) for w in words]
    result = []
    
    safe_punct = {'.', ',', '!', '?', ';', '\n'}
    
    i = 0
    while i < len(words):
        chunk_end = i + max_len
        
        if chunk_end >= len(words):
            chunk_idx = x_idx[i:]
            chunk_words = words[i:]
            i = len(words)
        else:
            safe_cut = chunk_end
            for j in range(chunk_end - 1, i, -1):
                if words[j] in safe_punct:
                    safe_cut = j + 1
                    break
            
            chunk_idx = x_idx[i:safe_cut]
            chunk_words = words[i:safe_cut]
            i = safe_cut
            
        pad_length = max_len - len(chunk_idx)
        padded_chunk = chunk_idx + [w2i["PAD"]] * pad_length
        x_tensor = torch.tensor([padded_chunk], dtype=torch.long).to(device)
        
        with torch.no_grad():
            preds = model(x_tensor)[0]
            
        valid_length = len(chunk_words)
        for j in range(valid_length):
            tag = i2t[preds[j]]
            result.append((chunk_words[j], tag))
            
    return result

def predict_ner_batch(texts, model, w2i, i2t, max_len, device, batch_size=64):
    all_chunks_idx = []
    all_chunks_words = []
    text_indices = []
    
    safe_punct = {'.', ',', '!', '?', ';', '\n'}
    
    for text_idx, text in enumerate(texts):
        text = str(text).strip()
        text = re.sub(r'([:.,!?()])', r' \1 ', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        
        if not words:
            continue
            
        x_idx = [w2i.get(w, w2i["UNK"]) for w in words]
        
        i = 0
        while i < len(words):
            chunk_end = i + max_len
            if chunk_end >= len(words):
                chunk_idx = x_idx[i:]
                chunk_words = words[i:]
                i = len(words)
            else:
                safe_cut = chunk_end
                for j in range(chunk_end - 1, i, -1):
                    if words[j] in safe_punct:
                        safe_cut = j + 1
                        break
                chunk_idx = x_idx[i:safe_cut]
                chunk_words = words[i:safe_cut]
                i = safe_cut
                
            pad_length = max_len - len(chunk_idx)
            padded_chunk = chunk_idx + [w2i["PAD"]] * pad_length
            
            all_chunks_idx.append(padded_chunk)
            all_chunks_words.append(chunk_words)
            text_indices.append(text_idx)
            
    if not all_chunks_idx:
        return [[] for _ in texts]
        
    all_chunks_tensor = torch.tensor(all_chunks_idx, dtype=torch.long).to(device)
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(all_chunks_tensor), batch_size):
            batch_x = all_chunks_tensor[i:i+batch_size]
            batch_preds = model(batch_x)
            all_preds.extend(batch_preds)
            
    results = [[] for _ in texts]
    for chunk_words, chunk_preds, txt_idx in zip(all_chunks_words, all_preds, text_indices):
        valid_length = len(chunk_words)
        for j in range(valid_length):
            tag = i2t[chunk_preds[j]]
            results[txt_idx].append((chunk_words[j], tag))
            
    return results

def clean_unicode(text):
    if not text: return ""
    return str(text).replace('\u2028', ' ').replace('\u2029', ' ')

def parse_steam_checklist_perfect(text):
    if not text: 
        return ""
    
    lines = text.replace('\r', '').split('\n')
    output_parts = []
    
    header_regex = re.compile(r'^[-=~_]{2,}\s*[\{\[\(]?\s*(.*?)\s*[\}\]\)]?\s*[-=~_]{2,}$')
    
    checked_regex = re.compile(r'^[\s]*([☑✅✔️]|\[x\]|\[X\]|\(\+\))\s*(.*)$', re.IGNORECASE)
    unchecked_regex = re.compile(r'^[\s]*([☐⬛🔲⬜]|\[ \]|\[\]|\( \))\s*(.*)$')
    
    current_header = None
    current_items = []
    
    def flush_checklist():
        if current_items:
            joined_items = ", ".join(current_items)
            if current_header:
                output_parts.append(f"{current_header}: {joined_items},")
            else:
                output_parts.append(f"{joined_items},")
            current_items.clear()

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue
            
        header_match = header_regex.match(original_line)
        if header_match:
            flush_checklist() 
            current_header = header_match.group(1).strip()
            continue
            
        if unchecked_regex.match(original_line):
            continue
            
        checked_match = checked_regex.match(original_line)
        if checked_match:
            raw_val = checked_match.group(2).strip()
            clean_value = re.sub(r'[^a-zA-Z0-9\s.,!?\'\-/()$€£:]', '', raw_val).strip()
            if clean_value:
                current_items.append(clean_value)
            continue
            
        flush_checklist()
        current_header = None
        output_parts.append(original_line)
        
    flush_checklist()
    return " ".join(output_parts)

def clean_raw_steam_review(text):
    if not text: return ""
    
    bbcode_tags = r'/?(h[1-6]|b|u|i|strike|spoiler|noparse|hr|list|table|th|tr|td|url|img|\*)'
    bb_pattern = r'\[' + bbcode_tags + r'(?:=[^\]]+)?\]'
    text = re.sub(bb_pattern, ' ', text, flags=re.IGNORECASE)
    
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = parse_steam_checklist_perfect(text)
    
    text = clean_unicode(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_game_name(app_id):
    try:
        url = "http://store.steampowered.com/api/appdetails"
        params = {
            "appids": app_id,
            "l": "english",
            "cc": "us"
        }
        response = requests.get(url, params=params)
        data = response.json()
        if str(app_id) in data and data[str(app_id)]['success']:
            return data[str(app_id)]['data']['name']
        return f"UnknownGame_{app_id}"
    except: 
        return f"AppID_{app_id}"

def estimate_time(review_count):
    requests_needed = review_count / 100
    est_seconds = requests_needed * 0.8
    if requests_needed > 150:
        est_seconds += 300 
    
    mins, secs = divmod(int(est_seconds), 60)
    if mins > 0:
        return f"~{mins} menit {secs} detik"
    return f"~{secs} detik"

# ==========================================
# 4. TAMPILAN USER INTERFACE (UI) STREAMLIT
# ==========================================
st.set_page_config(page_title="Demo NER Tugas Akhir", layout="wide")

col_head_left, col_head_right = st.columns([1.5, 1])

with col_head_left:
    st.title("Demo Model NER Aspect-Based Sentiment Analysis")
    st.markdown("**RNN + CRF**")

with col_head_right:
    with st.container(border=True):
        st.markdown("📌 **Aspek yang Didukung (Positif / Negatif):**")
        
        col_list1, col_list2 = st.columns(2)
        
        with col_list1:
            st.markdown("""
            - **Gameplay** (mekanik, fitur, kontrol)
            - **Performance** (bug, FPS, lag)
            - **Graphics** (visual, animasi, art style)
            - **Story** (cerita, plot, karakter)
            """)
            
        with col_list2:
            st.markdown("""
            - **Audio** (soundtrack, voice acting, SFX)
            - **Price** (harga, DLC, worth)
            - **OOA** (cheat, mod, community, dll)
            """)

st.divider()

col_model, _ = st.columns([1, 2])
with col_model:
    model_choice = st.selectbox(
        "Pilih Arsitektur Model untuk Analisis:",
        ("BiLSTM", "BiGRU", "SimpleRNN")
    )

with st.spinner(f"Sedang memuat model {model_choice}..."):
    try:
        model, w2i, i2t, config, device = load_model(model_choice)
        st.success(f"Model {model_choice} siap digunakan!")
            
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

st.write("") 

tab1, tab2, tab3 = st.tabs([
    "🔍 Analisis Teks Tunggal", 
    "📥 Scrape Steam Reviews", 
    "📊 Analisis Batch (Upload JSON)"
])

# ==========================================
# TAB 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    if "review_input" not in st.session_state:
        st.session_state.review_input = ""

    def set_random_review():
        templates = [
            # 1. Gameplay, Graphics, Performance
            "The combat is incredibly fluid and satisfying, making every boss fight a thrill. Visually, the open world is stunning with great lighting and breathtaking scenery. However, the frame drops in crowded cities are an absolute nightmare. They really need to fix the optimization.",
            # 2. Story, Audio, Price
            "The narrative completely pulled me in from the start, making me care deeply for every character. Combined with a masterpiece of a soundtrack that perfectly fits the mood, it's an unforgettable emotional journey. Honestly, getting this for $15 on sale is an absolute steal!",
            # 3. Gameplay, OOA, Performance
            "You can spend hundreds of hours just building, crafting, and exploring without getting bored. The modding community is fantastic and adds so much replay value to the base game. But be warned, the loading times are atrocious and it crashes randomly to the desktop.",
            # 4. Graphics, Story, Gameplay
            "Visually, it looks like a next-gen movie with absolutely beautiful textures and character models. The plot twists kept me on the edge of my seat the whole time! Sadly, the stealth mechanics feel extremely clunky and outdated compared to modern standards.",
            # 5. Price, OOA, Audio
            "I simply cannot justify the $70 price tag for such a ridiculously short campaign. Furthermore, the developers pushed a really invasive anti-cheat update that acts like spyware and blocks Linux users. At least the gun sound effects and voice acting are punchy and realistic.",
            # 6. Gameplay, Performance, OOA
            "The competitive gunplay is top tier right now. However, server latency and rubberbanding completely ruin the ranked experience. Also, the devs need to do something about the aimbot hackers ruining every lobby.",
            # 7. Story, Graphics, Price
            "A short but deeply emotional journey with a beautiful message about letting go. The pixel art style is incredibly charming, vibrant, and nostalgic. For just 5 bucks, this indie gem is completely worth your money and time. Buy it!",
            # 8. Audio, Gameplay, OOA
            "The OST is an absolute banger and syncs perfectly with your attacks! Dodging and parrying to the beat feels incredibly rewarding once you learn the patterns. The devs really poured their heart into this, communicating with the player base constantly on Discord.",
            # 9. Performance, Graphics, Price
            "What an unoptimized mess. I'm getting barely 20 FPS on a high-end rig with frequent stuttering. Even on ultra settings, the environments look muddy, blurry, and low-res. Definitely refunding this garbage, it's a total scam at full price.",
            # 10. Story, Audio, Gameplay, OOA
            "The lore is incredibly deep, though the main quest pacing drags a bit in the middle. Voice acting is phenomenal across the board, bringing the NPCs to life. Combat gets a bit repetitive after 50 hours, but I really appreciate the mod support that lets me fix the terrible UI."
        ]
        st.session_state.review_input = random.choice(templates)

    user_input = st.text_area("Masukkan teks ulasan game di sini:", key="review_input", height=150)
    
    col_btn1, col_btn2, _ = st.columns([2, 2, 8])
    with col_btn1:
        analyze_clicked = st.button("Analisis Teks Tunggal", type="primary", use_container_width=True)
    with col_btn2:
        st.button("Contoh review", on_click=set_random_review, use_container_width=True)

    if analyze_clicked:
        if st.session_state.review_input:
            with st.spinner("Menganalisis..."):
                max_length = config.get("max_len", 64)
                
                cleaned_input = clean_raw_steam_review(st.session_state.review_input)
                predictions = predict_ner(cleaned_input, model, w2i, i2t, max_length, device)
                
                extracted_entities = extract_entities_robust(predictions)
                entity_counts = Counter()
                
                for ent in extracted_entities:
                    entity_counts[ent["label"]] += 1
                
                st.divider()
                col_kiri, col_kanan = st.columns([1, 1])
                
                with col_kiri:
                    with st.expander("Lihat Detail Analisis per Kata", expanded=True):
                        df_detail = pd.DataFrame(predictions, columns=["Kata (Token)", "Tag BILOU"])
                        st.dataframe(df_detail, hide_index=True, use_container_width=True)
                
                with col_kanan:
                    st.subheader("Ringkasan Hasil Ekstraksi")
                    if entity_counts:
                        df_summary = pd.DataFrame(
                            entity_counts.items(), 
                            columns=["Aspek & Sentimen", "Jumlah Ditemukan"]
                        )
                        st.dataframe(df_summary, hide_index=True, use_container_width=True)
                        
                        st.write("**Detail Frasa yang Diekstrak:**")
                        for ent in extracted_entities:
                            st.markdown(f"- **{ent['label']}**: `{ent['phrase']}`")
                    else:
                        st.info("Tidak ada aspek/sentimen khusus yang terdeteksi.")
        else:
            st.warning("Teks ulasan tidak boleh kosong!")


# ==========================================
# TAB 2: SCRAPE STEAM REVIEWS
# ==========================================
with tab2:
    st.markdown("Cek informasi jumlah *review* terlebih dahulu sebelum melakukan pengunduhan data JSON.")
    
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    
    col_app, _ = st.columns([1, 2])
    with col_app:
        app_id_input = st.text_input("Masukkan Steam App ID (Contoh: 949230 dan 3595270):", value="")
    
    if st.session_state.preview_data and str(st.session_state.preview_data['app_id']) != app_id_input:
        st.session_state.preview_data = None
        if 'download_data' in st.session_state:
            del st.session_state['download_data']
            
    if st.button("Cek Info Game & Review"):
        if app_id_input.isdigit():
            app_id = int(app_id_input)
            with st.spinner("Mengecek informasi game ke Steam..."):
                game_name = get_game_name(app_id)
                preview_url = f"https://store.steampowered.com/appreviews/{app_id}"
                prev_params = {"json": "1", "language": "english", "num_per_page": "0"}
                try:
                    res = requests.get(preview_url, params=prev_params).json()
                    total_reviews = res.get('query_summary', {}).get('total_reviews', 0)
                    
                    st.session_state.preview_data = {
                        "app_id": app_id,
                        "game_name": game_name,
                        "total_reviews": total_reviews
                    }
                    if 'download_data' in st.session_state:
                        del st.session_state['download_data']
                except Exception as e:
                    st.error(f"Gagal mengambil info review: {e}")
        else:
            st.warning("App ID harus berupa angka.")

    if st.session_state.preview_data:
        data = st.session_state.preview_data
        st.success(f"**{data['game_name']}** terdeteksi! Total review berbahasa Inggris yang tersedia: **{data['total_reviews']:,}**")
        
        st.divider()
        st.write("### Opsi Pengunduhan")
        
        scrape_mode = st.radio("Pilih Mode Scraping:", ["Kustom (Tentukan Jumlah)", "Semua Review (Full Download)"])
        
        limit = data['total_reviews']
        if scrape_mode == "Kustom (Tentukan Jumlah)":
            limit = st.number_input(
                "Jumlah review yang ingin diunduh:", 
                min_value=1, 
                max_value=data['total_reviews'], 
                value=min(1000, data['total_reviews'])
            )
        
        st.info(f"⏳ Estimasi waktu scraping: **{estimate_time(limit)}**")
        
        if limit > 15000:
            st.warning("💡 **Catatan Sistem:** Steam membatasi penarikan data secara masif. Karena jumlah unduhan lebih dari 15.000 ulasan, sistem akan otomatis melakukan jeda (*cooldown*) selama 5 menit setiap kali kelipatan 15.000 ulasan tercapai untuk mencegah pemblokiran koneksi.")
        
        if st.button("Mulai Scraping Sekarang", type="primary"):
            app_id = data['app_id']
            game_title = data['game_name']
            processed_data = []
            
            if scrape_mode == "Semua Review (Full Download)":
                with st.spinner("Sedang mengambil semua data (bisa memakan waktu lama jika terkena cooldown)..."):
                    req_params = {"json": "1", "language": "english", "num_per_page": "100", "playtime_filter_min": "2"}
                    review_dict, query_count = steamreviews.download_reviews_for_app_id(
                        app_id, chosen_request_params=req_params
                    )
                    
                    if review_dict and 'reviews' in review_dict:
                        for review_id, review_data in review_dict["reviews"].items():
                            item = {
                                "review_id": review_id,
                                "app_id": app_id,
                                "game_name": game_title,
                                "steamid": review_data["author"]["steamid"],
                                "playtime_forever": review_data["author"]["playtime_forever"],
                                "review": clean_raw_steam_review(review_data["review"]),
                                "voted_up": review_data["voted_up"]
                            }
                            processed_data.append(item)
                            
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                cursor = "*"
                seen_review_ids = set()
                req_count = 0

                while len(processed_data) < limit:
                    params = {
                        "json": 1,
                        "language": "english",
                        "num_per_page": 100,
                        "playtime_filter_min": 2,
                        "filter": "recent",
                        "cursor": cursor
                    }
                    
                    try:
                        res = requests.get(f"https://store.steampowered.com/appreviews/{app_id}", params=params)
                        req_count += 1
                        
                        if res.status_code == 429 or req_count == 150:
                            status_text.warning("Tercapai batas akses Steam. Memulai cooldown 5 menit untuk mencegah blokir IP...")
                            
                            for sec_remaining in range(300, 0, -1):
                                mins, secs = divmod(sec_remaining, 60)
                                status_text.warning(f"⏳ Cooldown aktif... Lanjut otomatis dalam {mins:02d}:{secs:02d}")
                                time.sleep(1)
                            
                            req_count = 0
                            status_text.text(f"Melanjutkan pengunduhan... {len(processed_data)}/{limit} review unik")
                            
                            if res.status_code == 429:
                                continue 
                        
                        if res.status_code != 200:
                            st.error(f"Gagal mengambil data dari Steam. Status Code: {res.status_code}")
                            break
                            
                        resp_data = res.json()
                        
                        if "reviews" in resp_data and resp_data["reviews"]:
                            for review_data in resp_data["reviews"]:
                                rev_id = review_data["recommendationid"]
                                
                                if rev_id not in seen_review_ids:
                                    if len(processed_data) >= limit:
                                        break
                                    
                                    seen_review_ids.add(rev_id)
                                    
                                    item = {
                                        "review_id": rev_id,
                                        "app_id": app_id,
                                        "game_name": game_title,
                                        "steamid": review_data["author"]["steamid"],
                                        "playtime_forever": review_data["author"]["playtime_forever"],
                                        "review": clean_raw_steam_review(review_data["review"]),
                                        "voted_up": review_data["voted_up"]
                                    }
                                    processed_data.append(item)
                            
                            new_cursor = resp_data.get("cursor", cursor)

                            if new_cursor == cursor:
                                break
                                
                            cursor = new_cursor
                            
                            current_count = len(processed_data)
                            progress_bar.progress(min(current_count / limit, 1.0))
                            status_text.text(f"Mengunduh... {current_count}/{limit} review unik")
                            
                            time.sleep(0.5)
                        else:
                            break
                            
                    except Exception as e:
                        st.error(f"Koneksi terputus: {e}")
                        break
                progress_bar.empty()

            if processed_data:
                jsonl_str = "\n".join([json.dumps(x, ensure_ascii=False) for x in processed_data])
                st.session_state.download_data = jsonl_str
                st.session_state.download_filename = f"{app_id}_{game_title}_reviews.jsonl"
            else:
                st.warning("Tidak ada review yang ditemukan.")

        if 'download_data' in st.session_state:
            st.success("File siap diunduh!")
            st.download_button(
                label="📥 Unduh File JSONL",
                data=st.session_state.download_data,
                file_name=st.session_state.download_filename,
                mime="application/json"
            )


# ==========================================
# TAB 3: ANALISIS BATCH DARI JSON / JSONL
# ==========================================
with tab3:
    st.markdown("Unggah *file* `.json` atau `.jsonl` yang berisi data ulasan untuk dianalisis sekaligus.")
    
    uploaded_files = st.file_uploader("Pilih file JSON/JSONL", type=['json', 'jsonl'], accept_multiple_files=True)
    
    if uploaded_files:
        grouped_reviews = {}
        total_reviews_read = 0
        
        try:
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue().decode("utf-8")
                
                data_list = []
                try:
                    data_list = json.loads(file_content)
                    if isinstance(data_list, dict):
                        data_list = [data_list]
                except json.JSONDecodeError:
                    for line in file_content.strip().split('\n'):
                        if line.strip():
                            data_list.append(json.loads(line))
                            
                for data in data_list:
                    if "review" in data:
                        game_name = data.get("game_name", "Unknown Game")
                        if game_name not in grouped_reviews:
                            grouped_reviews[game_name] = []
                            
                        grouped_reviews[game_name].append(data)
                        total_reviews_read += 1
            
            st.success(f"Berhasil membaca total {total_reviews_read} review dari {len(uploaded_files)} file.")
            
            st.write("**Game yang terdeteksi:**")
            for game, revs in grouped_reviews.items():
                st.markdown(f"- **{game}**: {len(revs)} review")
            
            st.write("")

            if st.button("Mulai Analisis Batch", type="primary"):
                if total_reviews_read == 0:
                    st.error("Tidak ada data review yang valid untuk dianalisis.")
                    st.stop()
                    
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_progress = 0
                max_length = config.get("max_len", 64)
                
                for game_name, reviews in grouped_reviews.items():
                    
                    game_entity_counts = Counter()
                    game_analysis_results = []
                    
                    texts_to_process = [rev.get("review", "") for rev in reviews]
                    batch_predictions = []
                    
                    ui_chunk_size = 500
                    
                    for i in range(0, len(texts_to_process), ui_chunk_size):
                        sub_texts = texts_to_process[i : i + ui_chunk_size]
                        
                        sub_preds = predict_ner_batch(
                            sub_texts, model, w2i, i2t, max_length, device, batch_size=1024
                        )
                        batch_predictions.extend(sub_preds)
                        
                        current_progress += len(sub_texts)
                        progress_bar.progress(min(current_progress / total_reviews_read, 1.0))
                        status_text.text(f"Menganalisis {current_progress} dari {total_reviews_read} review...")
                    
                    for review_obj, preds in zip(reviews, batch_predictions):
                        extracted_entities = extract_entities_robust(preds)
                        
                        found_entities_str = []
                        for ent in extracted_entities:
                            base_label = ent["label"]
                            phrase = ent["phrase"]
                            
                            game_entity_counts[base_label] += 1
                            found_entities_str.append(f"{base_label} (\"{phrase}\")")
                        
                        game_analysis_results.append({
                            "Review ID": review_obj.get("review_id", "N/A"),
                            "Sentimen Umum (Steam)": "Positif" if review_obj.get("voted_up") else "Negatif",
                            "Teks Review": review_obj.get("review", ""),
                            "Detail Aspek Terdeteksi": " | ".join(found_entities_str) if found_entities_str else "None"
                        })
                        
                    
                    status_text.text(f"Menyiapkan visualisasi untuk {game_name}...")
                    col_batch_kiri, col_batch_kanan = st.columns([2, 1])
                    
                    with col_batch_kanan:
                        st.subheader(f"Total Aspek ({game_name})")
                        if game_entity_counts:
                            df_grand = pd.DataFrame(
                                game_entity_counts.items(), 
                                columns=["Aspek & Sentimen", "Total Frekuensi"]
                            )
                            df_grand = df_grand.sort_values(by="Total Frekuensi", ascending=False)
                            st.dataframe(df_grand, hide_index=True, use_container_width=True)
                            
                            fig = px.pie(
                                df_grand, 
                                names="Aspek & Sentimen", 
                                values="Total Frekuensi", 
                                hole=0.4,
                                title=f"Distribusi Analisis ({game_name})"
                            )

                            fig.update_traces(
                                hovertemplate="<b>%{label}</b><br>%{value}<extra></extra>"
                            )

                            fig.update_layout(
                                margin=dict(t=40, b=80, l=20, r=80)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Tidak ada entitas ditemukan.")
                    
                    with col_batch_kiri:
                        st.subheader(f"Analisis Game: {game_name}")
                        df_results = pd.DataFrame(game_analysis_results)
                        st.dataframe(df_results, hide_index=True, use_container_width=True)
                        
                        st.write("---") 
                        st.subheader("Highlight Analisis")
                        
                        if game_entity_counts:
                            total_frekuensi_aspek = sum(game_entity_counts.values())
                            aspek_terbanyak, frekuensi_terbanyak = game_entity_counts.most_common(1)[0]
                            
                            col_met1, col_met2, col_met3 = st.columns(3)
                            with col_met1:
                                st.metric(label="Total Ulasan Diproses", value=len(reviews))
                            with col_met2:
                                st.metric(label="Total Entitas Ditemukan", value=total_frekuensi_aspek)
                            with col_met3:
                                st.metric(
                                    label="Paling Banyak Dibahas", 
                                    value=aspek_terbanyak, 
                                    delta=f"{frekuensi_terbanyak} kali kemunculan",
                                    delta_color="off" 
                                )
                        else:
                            st.info("Belum ada entitas yang bisa disimpulkan dari ulasan ini.")
                        
                    st.divider() 
                    
                status_text.success("Analisis Batch Keseluruhan Selesai!")
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"Gagal membaca atau memproses file: {e}")