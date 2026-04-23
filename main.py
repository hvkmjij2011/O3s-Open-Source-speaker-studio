#!/usr/bin/python3.10
"""
Open-Source Speaker Studio — Tam Manuel Kontrol + Otomatik Config Üretici
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import soundfile as sf
from piper.voice import PiperVoice
import pedalboard
from pedalboard import (PitchShift, Distortion, Reverb,
                        Compressor, Gain, LowShelfFilter,
                        HighShelfFilter, PeakFilter, Chorus, NoiseGate)
import librosa
import os, json
import warnings
from PIL import Image, ImageTk
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_PATH   = "model.onnx"
CONFIG_PATH  = "model.onnx.json"
PRESETS_FILE = "presets.json"
LOGO_PATH    = "logo.png"

C = {
    "bg":    "#0a0a0f", 
    "panel": "#12121a", 
    "card":  "#181822",
    "border":"#252533", 
    "accent":"#0055ff", 
    "cyan":  "#4fc3f7",
    "success":"#4caf7d",
    "warn":  "#f0a500", 
    "danger":"#e05c5c",
    "pink":  "#f06292", 
    "text":  "#e8e8f0", 
    "dim":   "#7a7a96",
    "hover": "#1e1e2d",
}

DEFAULT_PRESETS = [
    ("Kerem",      "Genç Erkek",   C["cyan"],
     dict(pitch=-0.5, speed=1.08, reverb=0.03, distortion=0.0,
          noise_scale=0.35, noise_w=0.05,
          comp_threshold=-14.0, comp_ratio=2.5,
          comp_attack=5.0, comp_release=100.0,
          low_shelf_gain=1.5,  high_shelf_gain=0.5,  peak_gain=0.0,
          mid_freq=1000.0, stereo_width=0.0, pre_gain=0.0, post_gain=2.0))
]

SLIDER_DEFS = [
    # key, label, sublabel, min, max, default, color, group
    ("pitch",          "Pitch",             "eksi=kalın  artı=tiz",          -10.0, 10.0,  0.0,  C["accent"],  "TTS"),
    ("speed",          "Hız",               "0.5=yavaş  2.0=hızlı",           0.5,  2.0,   1.0,  C["cyan"],    "TTS"),
    ("noise_scale",    "Ses Varyasyonu",    "düşük=tutarlı  yüksek=doğal",    0.0,  1.0,   0.30, C["accent"],  "TTS"),
    ("noise_w",        "Ritim Varyasyonu",  "düşük=sabit  yüksek=akıcı",      0.0,  1.0,   0.00, C["accent"],  "TTS"),
    ("pre_gain",       "Giriş Gain",        "pedalboard öncesi seviye (dB)", -12.0, 12.0,  0.0,  C["cyan"],    "EFEKT"),
    ("distortion",     "Distortion",        "0=temiz  30=hırıltılı",           0.0, 30.0,  0.0,  C["warn"],    "EFEKT"),
    ("reverb",         "Reverb",            "0=kuru  1=büyük salon",           0.0,  0.5,  0.03, C["cyan"],    "EFEKT"),
    ("stereo_width",   "Stereo Genişlik",   "0=mono  1=geniş stereo",          0.0,  1.0,  0.0,  C["cyan"],    "EFEKT"),
    ("comp_threshold", "Comp Threshold",    "dB — sıkıştırma başlangıcı",    -40.0,  0.0,-12.0,  C["cyan"],    "KOMPRESÖR"),
    ("comp_ratio",     "Comp Ratio",        "1=yok  10=sert",                  1.0, 10.0,  3.0,  C["cyan"],    "KOMPRESÖR"),
    ("comp_attack",    "Comp Attack",       "ms — tepki hızı (hızlı=sert)",    1.0, 50.0,  5.0,  C["cyan"],    "KOMPRESÖR"),
    ("comp_release",   "Comp Release",      "ms — bırakma süresi",            20.0,500.0,100.0,  C["cyan"],    "KOMPRESÖR"),
    ("low_shelf_gain", "EQ Low Shelf",      "bas frekanslar (<300 Hz)",       -12.0, 12.0,  0.0, C["success"], "EQ"),
    ("mid_freq",       "EQ Mid Frekans",    "peak merkez Hz (300–5000)",      300.0,5000.0,1000.0,C["success"],"EQ"),
    ("peak_gain",      "EQ Mid Peak",       "mid frekans gain (dB)",          -12.0, 12.0,  0.0, C["success"], "EQ"),
    ("high_shelf_gain","EQ High Shelf",     "tiz frekanslar (>4000 Hz)",      -12.0, 12.0,  0.0, C["success"], "EQ"),
    ("post_gain",      "Çıkış Gain",        "son seviye artışı (dB)",          -6.0, 12.0,  2.0, C["success"], "EQ"),
]


class ModernSlider(tk.Frame):
    def __init__(self, parent, key, label, sublabel, from_, to, default, accent, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self.key = key
        row = tk.Frame(self, bg=C["card"])
        row.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(row, text=label, bg=C["card"], fg=C["text"],
                 font=("Helvetica", 9, "bold")).pack(side="left")
        self.val_lbl = tk.Label(row, text=f"{default:.2f}", bg=C["card"],
                                fg=accent, font=("Courier", 9, "bold"), width=7)
        self.val_lbl.pack(side="right")
        if sublabel:
            tk.Label(self, text=sublabel, bg=C["card"], fg=C["dim"],
                     font=("Helvetica", 7)).pack(anchor="w", padx=10)
        self.var = tk.DoubleVar(value=default)
        tk.Scale(self, variable=self.var, from_=from_, to=to,
                 orient="horizontal", resolution=(to-from_)/400,
                 showvalue=False, bg=C["card"], highlightthickness=0,
                 troughcolor=C["border"], activebackground=accent,
                 sliderrelief="flat", bd=0).pack(fill="x", padx=8, pady=(2,6))
        self.var.trace_add("write",
            lambda *_: self.val_lbl.config(text=f"{self.var.get():.2f}"))
    def get(self): return self.var.get()
    def set(self, v): self.var.set(float(v))


class PresetCard(tk.Frame):
    def __init__(self, parent, name, sublabel, color, on_click, on_delete, **kw):
        super().__init__(parent, bg=C["card"], cursor="hand2",
                         highlightbackground=C["border"], highlightthickness=1, **kw)
        self._cb = on_click
        row = tk.Frame(self, bg=C["card"])
        row.pack(fill="x", padx=8, pady=6)
        tk.Label(row, text="●", bg=C["card"], fg=color,
                 font=("Helvetica", 8)).pack(side="left", padx=(0,6))
        info = tk.Frame(row, bg=C["card"])
        info.pack(side="left", fill="x", expand=True)
        self._nl = tk.Label(info, text=name, bg=C["card"], fg=C["text"],
                            font=("Helvetica", 9, "bold"), anchor="w")
        self._nl.pack(fill="x")
        self._sl = tk.Label(info, text=sublabel, bg=C["card"], fg=C["dim"],
                            font=("Helvetica", 7), anchor="w")
        self._sl.pack(fill="x")
        del_btn = tk.Label(row, text="✕", bg=C["card"], fg=C["dim"],
                           font=("Helvetica", 9), cursor="hand2")
        del_btn.pack(side="right", padx=4)
        del_btn.bind("<Button-1>", lambda _: on_delete())
        for w in [self, row, info, self._nl, self._sl]:
            w.bind("<Enter>",    lambda _: self._bg(C["hover"]))
            w.bind("<Leave>",    lambda _: self._bg(C["card"]))
            w.bind("<Button-1>", lambda _: self._cb())

    def _bg(self, c):
        def _set(w):
            try: w.config(bg=c)
            except: pass
            for ch in w.winfo_children(): _set(ch)
        _set(self)

    def set_active(self, v):
        self.config(highlightbackground=C["accent"] if v else C["border"])


class PiperStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Open-Source Speaker Studio")
        self.root.geometry("1280x740")
        self.root.configure(bg=C["bg"])
        self.active_idx   = None
        self.preset_cards = []
        self.presets      = []

        if not os.path.exists(MODEL_PATH):
            tk.Label(root, text=f"Model bulunamadı: {MODEL_PATH}",
                     bg=C["bg"], fg=C["danger"],
                     font=("Helvetica", 13)).pack(expand=True)
            return

        self.voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)
        self.voice.config.noise_scale = 0.30
        self.voice.config.noise_w     = 0.00
        self._build()
        self._load_presets()

    def _build(self):
        # Header Bar
        bar = tk.Frame(self.root, bg=C["panel"], height=55)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        LOGO_PATH = "logo.png"

        if os.path.exists(LOGO_PATH):
            try:
                raw_img = Image.open(LOGO_PATH)
                aspect = raw_img.width / raw_img.height
                img = raw_img.resize((int(35 * aspect), 35), Image.Resampling.LANCZOS)
                self.logo_header = ImageTk.PhotoImage(img)
                tk.Label(bar, image=self.logo_header, bg=C["panel"]).pack(side="left", padx=15)
            except:
                tk.Label(bar, text="O³s STUDIO", bg=C["panel"], fg=C["accent"], font=("Courier", 14, "bold")).pack(side="left", padx=20)
        else:
            tk.Label(bar, text="O³s STUDIO", bg=C["panel"], fg=C["accent"], font=("Courier", 14, "bold")).pack(side="left", padx=20)

        self.status_var = tk.StringVar(value="Sistem Hazır")
        tk.Label(bar, textvariable=self.status_var, bg=C["panel"], fg=C["dim"], font=("Helvetica", 9)).pack(side="right", padx=20)

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)

    def _build_left(self, parent):
        frame = tk.Frame(parent, bg=C["panel"], width=220)
        frame.pack(side="left", fill="y", padx=(10,4), pady=10)
        frame.pack_propagate(False)
        tk.Label(frame, text="KARAKTERler", bg=C["panel"], fg=C["dim"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=12, pady=(12,4))
        self._canvas = tk.Canvas(frame, bg=C["panel"], highlightthickness=0, bd=0)
        self._canvas.pack(fill="both", expand=True, padx=6)
        self._list_frame = tk.Frame(self._canvas, bg=C["panel"])
        self._canvas.create_window((0,0), window=self._list_frame, anchor="nw")
        self._list_frame.bind("<Configure>",
            lambda e: self._canvas.config(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind_all("<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1*(e.delta//120), "units"))
        tk.Frame(frame, bg=C["border"], height=1).pack(fill="x", padx=8, pady=6)
        tk.Button(frame, text="🎙  WAV'dan Config Üret",
                  command=self._auto_config,
                  bg=C["card"], fg=C["cyan"], activebackground=C["hover"],
                  activeforeground=C["cyan"], font=("Helvetica", 9, "bold"),
                  relief="flat", bd=0, pady=10, cursor="hand2").pack(fill="x", padx=8, pady=(0,8))

    def _build_center(self, parent):
        mid = tk.Frame(parent, bg=C["bg"])
        mid.pack(side="left", fill="both", expand=True, pady=10, padx=4)
        tk.Label(mid, text="METİN", bg=C["bg"], fg=C["dim"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=4)
        border = tk.Frame(mid, bg=C["accent"], padx=1, pady=1)
        border.pack(fill="both", expand=True, pady=(4,8))
        self.txt = tk.Text(border, font=("Courier", 12), bg=C["card"], fg=C["text"],
                           insertbackground=C["accent"], relief="flat", bd=0,
                           padx=12, pady=10, undo=True, wrap="word",
                           selectbackground=C["accent"], selectforeground="#fff")
        self.txt.pack(fill="both", expand=True)
        self.txt.insert("1.0", "Merhaba, bu ses sentezi stüdyosuna hoş geldiniz.")
        self.btn = tk.Button(mid, text="▶  SESİ ÜRET VE KAYDET",
                             command=self.generate, bg=C["accent"], fg="#fff",
                             activebackground="#6355d4", activeforeground="#fff",
                             font=("Helvetica", 10, "bold"), relief="flat", bd=0,
                             padx=20, pady=12, cursor="hand2")
        self.btn.pack(fill="x", pady=(0,4))
        self.progress = ttk.Progressbar(mid, mode="indeterminate")
        self.progress.pack(fill="x")
        self.info = tk.Label(mid, text="", bg=C["bg"], fg=C["success"],
                             font=("Helvetica", 8))
        self.info.pack(anchor="w", padx=4, pady=2)

    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=C["panel"], width=275)
        outer.pack(side="right", fill="y", padx=(4,10), pady=10)
        outer.pack_propagate(False)
        tk.Label(outer, text="PARAMETRELER", bg=C["panel"], fg=C["dim"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=12, pady=(12,4))
        canvas = tk.Canvas(outer, bg=C["panel"], highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True, padx=6)
        inner = tk.Frame(canvas, bg=C["panel"])
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self.sliders = {}
        prev_group = None
        for key, label, sub, fmin, fmax, default, color, grp in SLIDER_DEFS:
            if grp != prev_group:
                tk.Label(inner, text=f"── {grp} ──", bg=C["panel"], fg=C["dim"],
                         font=("Helvetica", 7, "bold")).pack(anchor="w", padx=8, pady=(8,2))
                prev_group = grp
            wrap = tk.Frame(inner, bg=C["card"],
                            highlightbackground=C["border"], highlightthickness=1)
            wrap.pack(fill="x", pady=2)
            sl = ModernSlider(wrap, key, label, sub, fmin, fmax, default, color)
            sl.pack(fill="x")
            self.sliders[key] = sl
        tk.Frame(outer, bg=C["border"], height=1).pack(fill="x", padx=8, pady=6)
        tk.Button(outer, text="＋  Mevcut Ayarı Kaydet",
                  command=self._save_current_as_preset,
                  bg=C["card"], fg=C["success"], activebackground=C["hover"],
                  activeforeground=C["success"], font=("Helvetica", 9, "bold"),
                  relief="flat", bd=0, pady=10, cursor="hand2").pack(fill="x", padx=8, pady=(0,8))

    # ── Preset Yönetimi ──────────────────────────────────────────────────────

    def _load_presets(self):
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                    self.presets = json.load(f)
            except Exception:
                self.presets = []
        if not self.presets:
            self.presets = [{"name":n,"sublabel":s,"color":col,"params":p}
                            for n,s,col,p in DEFAULT_PRESETS]
        self._refresh_list()

    def _save_presets_file(self):
        try:
            with open(PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Kayıt Hatası", str(e))

    def _refresh_list(self):
        for w in self._list_frame.winfo_children():
            w.destroy()
        self.preset_cards = []
        for i, p in enumerate(self.presets):
            card = PresetCard(
                self._list_frame, p["name"], p["sublabel"], p.get("color", C["accent"]),
                on_click=lambda _i=i: self._apply_preset(_i),
                on_delete=lambda _i=i: self._delete_preset(_i),
            )
            card.pack(fill="x", pady=3)
            self.preset_cards.append(card)
        if self.active_idx is not None and self.active_idx < len(self.preset_cards):
            self.preset_cards[self.active_idx].set_active(True)

    def _apply_preset(self, idx):
        p = self.presets[idx]["params"]
        for key, sl in self.sliders.items():
            if key in p: sl.set(p[key])
        if self.active_idx is not None and self.active_idx < len(self.preset_cards):
            self.preset_cards[self.active_idx].set_active(False)
        self.preset_cards[idx].set_active(True)
        self.active_idx = idx
        self.status_var.set(f"Preset: {self.presets[idx]['name']}")

    def _delete_preset(self, idx):
        name = self.presets[idx]["name"]
        if not messagebox.askyesno("Sil", f'"{name}" silinsin mi?'): return
        self.presets.pop(idx)
        if self.active_idx == idx: self.active_idx = None
        elif self.active_idx and self.active_idx > idx: self.active_idx -= 1
        self._save_presets_file()
        self._refresh_list()

    def _save_current_as_preset(self):
        name = simpledialog.askstring("Preset İsmi", "Bu karakter için bir isim gir:",
                                      parent=self.root)
        if not name or not name.strip(): return
        sublabel = simpledialog.askstring("Alt Etiket", "Kısa açıklama (örn: Genç Erkek):",
                                          parent=self.root) or ""
        params = {key: sl.get() for key, sl in self.sliders.items()}
        colors = [C["accent"],C["cyan"],C["pink"],C["warn"],C["success"],C["danger"]]
        color  = colors[len(self.presets) % len(colors)]
        self.presets.append({"name":name.strip(),"sublabel":sublabel.strip(),
                              "color":color,"params":params})
        self._save_presets_file()
        self._refresh_list()
        self._apply_preset(len(self.presets)-1)
        self.status_var.set(f"✓ '{name.strip()}' kaydedildi")

    # ── WAV → Otomatik Config ────────────────────────────────────────────────

    def _auto_config(self):
        # 1) Referans WAV seç
        path = filedialog.askopenfilename(title="Referans WAV seç (kendi sesin)",
                                          filetypes=[("WAV","*.wav"),("Tüm","*.*")])
        if not path: return

        # 2) WAV'da okunan metni sor — olmadan speed/noise karşılaştırması yapılamaz
        ref_text = simpledialog.askstring(
            "Referans Metin",
            "WAV dosyasında okunan metni gir:\n(Boş bırakırsan sadece pitch ve EQ analiz edilir)",
            parent=self.root) or ""

        self.status_var.set("Analiz ediliyor..."); self.root.update()
        try:
            params = self._analyze_wav(path, ref_text.strip())
        except Exception as e:
            messagebox.showerror("Analiz Hatası", str(e))
            self.status_var.set("Hata"); return

        for key, sl in self.sliders.items():
            if key in params: sl.set(params[key])
        self.status_var.set("Analiz tamamlandı")

        name = simpledialog.askstring("Preset İsmi", "Bu WAV için bir isim gir:",
                                      parent=self.root)
        if name and name.strip():
            sublabel = simpledialog.askstring("Alt Etiket", "Kısa açıklama:",
                                              parent=self.root) or "WAV analizi"
            colors = [C["accent"],C["cyan"],C["pink"],C["warn"],C["success"]]
            color  = colors[len(self.presets) % len(colors)]
            self.presets.append({"name":name.strip(),"sublabel":sublabel.strip(),
                                  "color":color,"params":params})
            self._save_presets_file()
            self._refresh_list()
            self._apply_preset(len(self.presets)-1)
            self.status_var.set(f"✓ '{name.strip()}' eklendi")

    def _analyze_wav(self, path: str, ref_text: str = "") -> dict:
        SR = 22050

        # ── Yükle + sessizlik kırp + normalize ─────────────────────────────
        y_raw, _ = librosa.load(path, sr=SR, mono=True)
        y_ref, _ = librosa.effects.trim(y_raw, top_db=25)
        y_ref    = librosa.util.normalize(y_ref)

        # ── PITCH — F0 medyanı → Piper baz 120 Hz → semitone ───────────────
        f0, voiced_flag, _ = librosa.pyin(y_ref, fmin=50, fmax=500, sr=SR)
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        mean_f0   = float(np.median(f0_voiced)) if len(f0_voiced) > 0 else 120.0
        pitch_st  = float(np.clip(12 * np.log2(mean_f0 / 120.0), -10, 10))

        # ── SPEED — metin varsa Piper gerçek çıktısıyla karşılaştır ─────────
        if ref_text:
            self.voice.config.length_scale = 1.0
            self.voice.config.noise_scale  = 0.30
            self.voice.config.noise_w      = 0.00
            piper_chunks = []
            for chunk in self.voice.synthesize(ref_text):
                if hasattr(chunk, "audio_int16_bytes"):
                    piper_chunks.append(chunk.audio_int16_bytes)
            if piper_chunks:
                y_piper = (np.frombuffer(b"".join(piper_chunks), dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                # Hece tespiti — backtrack=True daha doğru onset
                def syllables_per_sec(y):
                    onsets = librosa.onset.onset_detect(
                        y=y, sr=SR, units="time", backtrack=True,
                        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.07, wait=10)
                    return len(onsets) / (len(y) / SR + 1e-9)
                sps_ref   = syllables_per_sec(y_ref)
                sps_piper = syllables_per_sec(y_piper)
                speed = float(np.clip(sps_ref / (sps_piper + 1e-9), 0.5, 2.0))
            else:
                speed = 1.0
        else:
            # Metin yok — onset yoğunluğu + normalleştirilmiş Türkçe tempo (4.5 syl/s)
            onsets = librosa.onset.onset_detect(y=y_ref, sr=SR, units="time",
                                                pre_max=3, post_max=3, pre_avg=3,
                                                post_avg=5, delta=0.07, wait=10)
            sps   = len(onsets) / (len(y_ref) / SR + 1e-9)
            speed = float(np.clip(sps / 4.5, 0.5, 2.0))

        # ── NOISE_SCALE — F0 jitter (perde titremesi = doğallık) ────────────
        noise_scale = float(np.clip(np.std(f0_voiced) / 30.0, 0.10, 0.80)) \
                      if len(f0_voiced) > 1 else 0.30

        # ── NOISE_W — onset zamanlama düzensizliği = ritim varyasyonu ────────
        onsets_ref = librosa.onset.onset_detect(y=y_ref, sr=SR, units="time",
                                                pre_max=3, post_max=3, pre_avg=3,
                                                post_avg=5, delta=0.07, wait=10)
        if len(onsets_ref) > 2:
            iv      = np.diff(onsets_ref)
            noise_w = float(np.clip(np.std(iv) / (np.mean(iv) + 1e-9), 0.0, 0.40))
        else:
            noise_w = 0.05

        # ── MEL Spektrum — EQ tabanı ─────────────────────────────────────────
        mel    = librosa.feature.melspectrogram(y=y_ref, sr=SR, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        bass_db = float(np.mean(mel_db[0:20, :]))    # ~80–500 Hz
        mid_db  = float(np.mean(mel_db[20:80, :]))   # ~500 Hz–4 kHz  ← referans
        high_db = float(np.mean(mel_db[80:,   :]))   # ~4–8 kHz

        low_shelf_gain  = float(np.clip((bass_db - mid_db) * 0.35, -8.0, 8.0))
        high_shelf_gain = float(np.clip((high_db - mid_db) * 0.35, -8.0, 8.0))

        # ── MID_FREQ — Spectral Centroid + LPC formant doğrulaması ──────────
        # Önce spectral centroid ile merkez frekansı bul
        sc = float(np.mean(librosa.feature.spectral_centroid(y=y_ref, sr=SR)))
        # LPC ile ilk 2 formantı tahmin et (konuşma sesi için F1/F2 karakteristik)
        try:
            # LPC için pre-emphasis + pencere
            y_emph = np.append(y_ref[0], y_ref[1:] - 0.97 * y_ref[:-1])
            lpc_order = int(SR / 1000) + 2   # örn. 22050 → ~24
            # Otokorelasyon tabanlı LPC (Levinson-Durbin)
            frame_len = 2048
            hop = 512
            formants = []
            for i in range(0, len(y_emph) - frame_len, hop * 4):
                frame = y_emph[i:i+frame_len] * np.hanning(frame_len)
                # LPC çöz
                r = np.correlate(frame, frame, mode='full')
                r = r[len(r)//2:]
                R = np.array([[r[abs(i-j)] for j in range(lpc_order)] for i in range(lpc_order)])
                try:
                    a = np.linalg.solve(R, -r[1:lpc_order+1])
                    a = np.concatenate([[1], a])
                    # Kökler → frekanslar
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) >= 0]
                    angles = np.angle(roots)
                    freqs  = angles * SR / (2 * np.pi)
                    freqs  = freqs[(freqs > 200) & (freqs < 4000)]
                    if len(freqs):
                        formants.append(np.min(freqs))  # F1 ≈ en düşük
                except Exception:
                    pass
            if formants:
                f1_mean = float(np.median(formants))
                # mid_freq: F1 ile spectral centroid ortalaması, 300–5000 Hz arası
                mid_freq = float(np.clip((f1_mean + sc) / 2.0, 300.0, 5000.0))
            else:
                mid_freq = float(np.clip(sc * 0.7, 300.0, 5000.0))
        except Exception:
            mid_freq = float(np.clip(sc * 0.7, 300.0, 5000.0))

        # ── PEAK_GAIN — mid band asimetri ───────────────────────────────────
        # mel_db'de mid_freq'e en yakın mel bandını bul
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmax=8000)
        mid_band  = int(np.argmin(np.abs(mel_freqs - mid_freq)))
        lo_band   = max(0, mid_band - 15)
        hi_band   = min(127, mid_band + 15)
        mid_lo_db = float(np.mean(mel_db[lo_band:mid_band, :]))
        mid_hi_db = float(np.mean(mel_db[mid_band:hi_band, :]))
        peak_gain = float(np.clip((mid_lo_db - mid_hi_db) * 0.30, -6.0, 6.0))

        # ── REVERB — rolloff spread ──────────────────────────────────────────
        r85    = float(np.mean(librosa.feature.spectral_rolloff(y=y_ref, sr=SR, roll_percent=0.85)))
        r50    = float(np.mean(librosa.feature.spectral_rolloff(y=y_ref, sr=SR, roll_percent=0.50)))
        reverb = float(np.clip((r85 - r50) / SR * 0.8, 0.0, 0.20))

        # ── DISTORTION — spectral flatness (0=temiz tonlu, 1=gürültülü) ────
        flatness   = float(np.mean(librosa.feature.spectral_flatness(y=y_ref)))
        distortion = float(np.clip(flatness * 30.0, 0.0, 10.0))

        # ── COMPRESSOR — crest factor + dinamik analiz ────────────────────
        rms   = float(np.sqrt(np.mean(y_ref ** 2)))
        peak  = float(np.max(np.abs(y_ref)))
        crest = float(20 * np.log10(peak / (rms + 1e-9)))
        comp_threshold = float(np.clip(-crest,       -30.0, -6.0))
        comp_ratio     = float(np.clip(crest / 6.0,   1.5,   6.0))

        # comp_attack — kurtosis bazlı: yüksek kurtosis (sivri geçiciler) → hızlı attack
        # Analiz sonucu: orijinal kurtosis=7.28, sentez=0.43 → büyük fark
        # Referans ses yüksek kurtosis → sert geçiciler → kısa attack gerekir
        from scipy.stats import kurtosis as sp_kurt
        kurt = float(sp_kurt(y_ref))
        comp_attack  = float(np.clip(20.0 / (1.0 + max(kurt, 0.1)), 1.0, 30.0))
        # comp_release — pause ratio ile orantılı: uzun duraklar → uzun release
        pause_ratio = 1.0 - (float(np.sum(voiced_flag)) / len(voiced_flag))
        comp_release = float(np.clip(50.0 + pause_ratio * 300.0, 30.0, 400.0))

        # ── STEREO WIDTH — varsayılan 0 (mono TTS için) ─────────────────────
        stereo_width = 0.0

        # ── PRE/POST GAIN — RMS hedef seviyesi ──────────────────────────────
        # Hedef RMS: -18 dBFS civarı (yayın standardı)
        target_rms_db = -18.0
        current_rms_db = 20 * np.log10(rms + 1e-9)
        pre_gain  = 0.0  # Kullanıcı tercihi, analiz atamasına gerek yok
        post_gain = float(np.clip(target_rms_db - current_rms_db, -6.0, 12.0))

        return dict(
            pitch=round(pitch_st, 2),
            speed=round(speed, 2),
            noise_scale=round(noise_scale, 2),
            noise_w=round(noise_w, 3),
            reverb=round(reverb, 3),
            distortion=round(distortion, 1),
            comp_threshold=round(comp_threshold, 1),
            comp_ratio=round(comp_ratio, 2),
            comp_attack=round(comp_attack, 1),
            comp_release=round(comp_release, 0),
            low_shelf_gain=round(low_shelf_gain, 1),
            high_shelf_gain=round(high_shelf_gain, 1),
            peak_gain=round(peak_gain, 2),
            mid_freq=round(mid_freq, 0),
            stereo_width=stereo_width,
            pre_gain=round(pre_gain, 1),
            post_gain=round(post_gain, 1),
        )

    # ── Ses Üretimi ──────────────────────────────────────────────────────────

    def generate(self):
        text = self.txt.get("1.0", "end-1c").strip()
        if not text: return
        self.btn.config(state="disabled", text="⏳  İşleniyor...")
        self.info.config(text=""); self.progress.start(12); self.root.update()
        try:
            s = {k: sl.get() for k, sl in self.sliders.items()}
            self.voice.config.length_scale = 1.0 / s["speed"]
            self.voice.config.noise_scale  = s["noise_scale"]
            self.voice.config.noise_w      = s["noise_w"]

            chunks = []
            for chunk in self.voice.synthesize(text):
                if hasattr(chunk, "audio_int16_bytes"):
                    chunks.append(chunk.audio_int16_bytes)
            if not chunks:
                self._msg("❌ Ses verisi boş!", C["danger"]); return

            audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0

            board = pedalboard.Pedalboard([
                PitchShift(semitones=s["pitch"]),
                Distortion(drive_db=s["distortion"]) if s["distortion"] > 0 else Gain(gain_db=0),
                LowShelfFilter(cutoff_frequency_hz=300,  gain_db=s["low_shelf_gain"],  q=0.7),
                PeakFilter(cutoff_frequency_hz=1000,     gain_db=s["peak_gain"],       q=1.0),
                HighShelfFilter(cutoff_frequency_hz=4000,gain_db=s["high_shelf_gain"], q=0.7),
                Reverb(room_size=s["reverb"], wet_level=0.15, dry_level=0.85),
                Compressor(threshold_db=s["comp_threshold"], ratio=s["comp_ratio"]),
                Gain(gain_db=2.0),
            ])

            out = board(audio, self.voice.config.sample_rate)
            sf.write("output.wav", out, self.voice.config.sample_rate)
            dur = len(out) / self.voice.config.sample_rate
            self._msg(f"✓  output.wav kaydedildi  ({dur:.1f}s)", C["success"])

        except Exception as e:
            self._msg(f"❌ {e}", C["danger"])
        finally:
            self.progress.stop()
            self.btn.config(state="normal", text="▶  SESİ ÜRET VE KAYDET")

    def _msg(self, text, color):
        self.info.config(text=text, fg=color)
        self.status_var.set(text[:70])


if __name__ == "__main__":
    root = tk.Tk()
    try:
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TProgressbar", troughcolor=C["panel"], background=C["accent"])
    except Exception:
        pass
    PiperStudio(root)
    root.mainloop()
