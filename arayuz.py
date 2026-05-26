import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from mamba_lens import HookedMamba, MambaCfg

# Sayfa Yapılandırması (Modern ve Profesyonel Görünüm)
st.set_page_config(
    page_title="Mamba Kestirimci Bakım Paneli",
    page_icon="⚙️",
    layout="centered"
)

# =====================================================================
# 1. MODEL MİMARİSİ (Eğitim koduyla birebir aynı olmak zorunda!)
# =====================================================================
class TabularMambaModel(nn.Module):
    def __init__(self, d_model=16, d_state=16, d_conv=4, expand=2, num_layers=2):
        super().__init__()
        self.type_emb = nn.Embedding(num_embeddings=3, embedding_dim=d_model)
        self.continuous_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(5)])
        
        self.cfg = MambaCfg(
            d_model=d_model,
            n_layers=num_layers,
            vocab_size=3,
            device='cpu'
        )
        # Eğitimde iç hataları bypass ettiğimiz yöntemle başlatıyoruz
        self.mamba = HookedMamba(self.cfg, tokenizer=None, initialize_params=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(6 * d_model, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        type_indices = x[:, 0].long()
        type_embedded = self.type_emb(type_indices).unsqueeze(1)
        
        continuous_embeddings = []
        for i in range(5):
            col = x[:, i+1 : i+2]
            continuous_embeddings.append(self.continuous_projs[i](col))
            
        continuous_embedded = torch.stack(continuous_embeddings, dim=1)
        x_seq = torch.cat([type_embedded, continuous_embedded], dim=1)
        
        # Mamba bloklarından geçiş
        for layer in self.mamba.blocks:
            x_seq = layer(x_seq)
            
        x_seq = self.mamba.norm(x_seq)
        x_flat = x_seq.reshape(x_seq.size(0), -1)
        logits = self.classifier(x_flat)
        return logits

# =====================================================================
# 2. MODEL VE SCALER NESNELERİNİN YÜKLENMESİ
# =====================================================================
@st.cache_resource
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMambaModel()
    
    try:
        # Kaydettiğin kütüphaneli ağırlıkları yüklüyoruz
        model.load_state_dict(torch.load("mamba_lens_model_weights.pth", map_location=device))
        model.to(device)
        model.eval()
        
        # Eğitimdeki gerçek StandardScaler nesneni yüklüyoruz
        scaler = joblib.load("scaler.pkl")
        
        return model, scaler, device
    except FileNotFoundError as e:
        st.error(f"🚨 Gerekli dosya bulunamadı: {e.filename}. Lütfen dosyanın bu scriptle aynı klasörde olduğundan emin olun.")
        return None, None, device

model, scaler, device = load_artifacts()

# =====================================================================
# 3. KULLANICI ARAYÜZÜ TASARIMI (Görsel Alan)
# =====================================================================
st.title("⚙️ Mamba Kestirimci Bakım Paneli")
st.markdown("Bu panel, fabrikanızdaki makinelerin sensör verilerini **Mamba (State Space Model)** mimarisiyle analiz ederek olası arıza risklerini anlık olarak tahmin eder.")
st.divider()

# Form Yapısı
with st.form("sensor_form"):
    st.subheader("📊 Anlık Sensör Veri Girişi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_type = st.selectbox("Makine Tipi (Type)", ["L (Low Quality)", "M (Medium Quality)", "H (High Quality)"])
        air_temp = st.number_input("Hava Sıcaklığı (Air temperature [K])", min_value=200.0, max_value=400.0, value=300.0, step=0.1)
        process_temp = st.number_input("İşlem Sıcaklığı (Process temperature [K])", min_value=200.0, max_value=400.0, value=310.0, step=0.1)
        
    with col2:
        rot_speed = st.number_input("Dönüş Hızı (Rotational speed [rpm])", min_value=500, max_value=3000, value=1500, step=1)
        torque = st.number_input("Tork Değeri (Torque [Nm])", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        tool_wear = st.number_input("Takım Aşınması (Tool wear [min])", min_value=0, max_value=300, value=100, step=1)
        
    submit_button = st.form_submit_button("🔮 Makine Durumunu Analiz Et")

# =====================================================================
# 4. TAHMİN MOTORU VE SİNYAL İŞLEME
# =====================================================================
if submit_button:
    if model is not None and scaler is not None:
        # 1. Kategorik veriyi (Type) eğitimdeki LabelEncoder sırasına göre eşliyoruz:
        # H -> 0, M -> 1, L -> 2
        type_mapping = {"H (High Quality)": 0, "M (Medium Quality)": 1, "L (Low Quality)": 2}
        type_val = type_mapping[machine_type]
        
        # 2. Sayısal verileri bir araya getirip senin 'scaler.pkl' nesnenle tam olarak aynı mantıkta ölçeklendiriyoruz
        raw_continuous = pd.DataFrame([{
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rot_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear
        }])
        
        scaled_continuous = scaler.transform(raw_continuous)[0]
        
        # 3. Model girdisini birleştiriyoruz [Type, Air, Process, Rot, Torque, Wear]
        input_array = np.insert(scaled_continuous, 0, type_val)
        input_tensor = torch.tensor([input_array], dtype=torch.float32).to(device)
        
        # 4. İleri Geçiş (Prediction)
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            
        st.divider()
        st.subheader("🎯 Model Analiz Sonucu")
        
        # Az önce yakaladığın o en dengeli, en efsane threshold değerini buraya koyuyoruz!
        # Hatırla, 0.75 yaptığında yanlış alarmlar çok azalmıştı.
        threshold = 0.75 
        
        if prob > threshold:
            st.error(f"🚨 **DİKKAT: ARIZA RİSKİ YÜKSEK!** (Kestirilen Olasılık: %{prob*100:.2f})")
            st.markdown("""
            👉 **Sistem Önerileri:**
            * Makinenin **Tork** ve **Dönüş Hızı** dengesini kontrol edin.
            * **Takım Aşınması (Tool wear)** sınır değerine yakın olabilir; kesici ucu yenileyin.
            * Operasyonu durdurup önleyici bakıma alın.
            """)
        else:
            st.success(f"✅ **SİSTEM STABİL: ARIZA RİSKİ YOK.** (Kestirilen Olasılık: %{prob*100:.2f})")
            st.markdown("👉 **Sistem Önerisi:** Mevcut sensör değerleri güvenli sınırlar içerisinde. Operasyona devam edilebilir.")