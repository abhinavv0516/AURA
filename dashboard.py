import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image, ImageTk
import numpy as np
import time

# ── Aesthetic Configuration (Japanese Lab) ─────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Colors
BG          = '#F4F6F9' # Warm off-white
CARD        = '#FFFFFF' # Pure white
PANEL       = '#FAFBFE' # Almost white with a blue tint
BORDER      = '#E2E6EF' # Barely-there hairlines
TEXT        = '#1A1D2E' # Near-black, soft
SUBTEXT     = '#6B7280' # Cool gray
ACCENT      = '#0057FF' # Deep royal blue
GREEN       = '#00C896' # Mint green
YELLOW      = '#F5A623' # Amber
RED         = '#E8294A' # Clean red
LOG_BG      = '#1A1D2E' # Dark contrast island for logs

# Fonts
# Note: Inter/DM Sans are specified in rcParams for matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Inter', 'DM Sans', 'Helvetica', 'Arial']

class Dashboard:
    def __init__(self, fusion_engine):
        self.fusion_engine = fusion_engine
        self.fusion_scores  = []
        self._temp_history  = []
        self._vibe_history  = []
        self._last_temp     = 35.0
        self.t_start        = time.time()

        # ── Root Window ──────────────────────────────────────────
        self.root = ctk.CTk()
        self.root.title("AURA — Motor Health Monitor | LIVE")
        self.root.geometry("1500x920")
        self.root.configure(fg_color=BG)

        # ── Top Header Bar ───────────────────────────────────────
        self.header = ctk.CTkFrame(self.root, height=56, fg_color=CARD, corner_radius=0, border_width=0)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)

        # Header Content
        self.logo_label = ctk.CTkLabel(self.header, text="⚡ AURA", font=("Inter", 24, "bold"), text_color=ACCENT)
        self.logo_label.pack(side=tk.LEFT, padx=(30, 5))
        
        self.subtitle_label = ctk.CTkLabel(self.header, text="Motor Health Monitor", font=("Inter", 14), text_color=SUBTEXT)
        self.subtitle_label.pack(side=tk.LEFT, pady=(5, 0))

        # Right side: Clock & Status
        self.clock_lbl = ctk.CTkLabel(self.header, text="", font=("JetBrains Mono", 12), text_color=TEXT)
        self.clock_lbl.pack(side=tk.RIGHT, padx=30)
        
        self.status_dot = ctk.CTkLabel(self.header, text="●", font=("Inter", 18), text_color=GREEN)
        self.status_dot.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.version_tag = ctk.CTkLabel(self.header, text="v2.4.1", font=("Inter", 10), text_color=BORDER)
        self.version_tag.pack(side=tk.RIGHT, padx=10)

        # Bottom Border Gradient Line
        self.grad_line = ctk.CTkFrame(self.root, height=2, fg_color=ACCENT, corner_radius=0)
        self.grad_line.pack(fill=tk.X)

        # ── Tab Navigation ───────────────────────────────────────
        self.tab_view = ctk.CTkTabview(self.root, fg_color=BG, segmented_button_fg_color=BG,
                                       segmented_button_selected_color=CARD,
                                       segmented_button_selected_hover_color=CARD,
                                       segmented_button_unselected_color=BG,
                                       text_color=SUBTEXT)
        self.tab_view.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        self.tab_hw = self.tab_view.add("📡  Live Hardware Feed")
        self.tab_ai = self.tab_view.add("🧠  AI Vision Demo")

        self._build_hardware_tab()
        self._build_ai_tab()

        # Initial clock tick
        self._tick_clock()
        self._animate_status_dot()

    def _build_hardware_tab(self):
        parent = self.tab_hw
        parent.grid_columnconfigure(1, weight=1)

        # LEFT COLUMN (30%)
        left_col = ctk.CTkFrame(parent, fg_color=BG)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Temperature Card
        self.temp_card = self._create_card(left_col, "TEMPERATURE SENSOR")
        self.temp_card.pack(fill=tk.X, pady=(0, 15))
        
        self.temp_val_lbl = ctk.CTkLabel(self.temp_card, text="29.0°C", font=("Inter", 48, "bold"), text_color=ACCENT)
        self.temp_val_lbl.pack(pady=(10, 0))
        
        self.hum_lbl = ctk.CTkLabel(self.temp_card, text="Humidity: 38.0%", font=("Inter", 12), text_color=SUBTEXT)
        self.hum_lbl.pack(pady=(0, 10))
        
        self.temp_status_pill = ctk.CTkLabel(self.temp_card, text="● NORMAL", font=("Inter", 12, "bold"), 
                                            fg_color="#E6F9F2", text_color=GREEN, corner_radius=20, 
                                            width=120, height=32)
        self.temp_status_pill.pack(pady=(0, 10))

        # Sensor Info Card
        self.info_card = self._create_card(left_col, "SYSTEM SPECIFICATIONS")
        self.info_card.pack(fill=tk.BOTH, expand=True)
        
        self._add_spec_row(self.info_card, "SENSOR", "DHT11")
        self._add_spec_row(self.info_card, "VIBRATION", "ADXL345")
        self._add_spec_row(self.info_card, "SAMPLING", "50 HZ")
        self._add_spec_row(self.info_card, "PROTOCOL", "UDP STREAM")

        # RIGHT COLUMN (70%)
        right_col = ctk.CTkFrame(parent, fg_color=BG)
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.grid_columnconfigure(0, weight=1)

        # Temperature History
        self.t_hist_card = self._create_card(right_col, "TEMPERATURE HISTORY (°C)")
        self.t_hist_card.pack(fill=tk.X, pady=(0, 15))
        
        self.fig_t, self.ax_t = plt.subplots(figsize=(8, 2.5), facecolor=CARD)
        self._style_matplotlib_ax(self.ax_t)
        self.canvas_t = FigureCanvasTkAgg(self.fig_t, master=self.t_hist_card)
        self.canvas_t.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bottom row for Vibration and Acoustic
        bot_row = ctk.CTkFrame(right_col, fg_color=BG)
        bot_row.pack(fill=tk.BOTH, expand=True)
        bot_row.grid_columnconfigure((0, 1), weight=1)

        # Vibration Card (Time Series)
        self.vib_card = self._create_card(bot_row, "VIBRATION LEVEL")
        self.vib_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        
        self.fig_v, self.ax_v = plt.subplots(figsize=(4, 3), facecolor=CARD)
        self._style_matplotlib_ax(self.ax_v)
        self.ax_v.set_ylim(230, 350) # Fixed Y-axis to accommodate +100 spikes
        self.canvas_v = FigureCanvasTkAgg(self.fig_v, master=self.vib_card)
        self.canvas_v.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Acoustic Card
        self.aco_card = self._create_card(bot_row, "ACOUSTIC WAVEFORM")
        self.aco_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        
        self.fig_a, self.ax_a = plt.subplots(figsize=(4, 3), facecolor=CARD)
        self._style_matplotlib_ax(self.ax_a)
        self.canvas_a = FigureCanvasTkAgg(self.fig_a, master=self.aco_card)
        self.canvas_a.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _build_ai_tab(self):
        parent = self.tab_ai
        parent.grid_columnconfigure(1, weight=1)

        # LEFT PANEL (40%)
        left_col = ctk.CTkFrame(parent, fg_color=BG)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Grad-CAM Feed
        self.vision_card = self._create_card(left_col, "AI GRAD-CAM XAI FEED")
        self.vision_card.pack(fill=tk.X, pady=(0, 15))
        
        self.video_label = tk.Label(self.vision_card, bg="#F0F0F0", width=520, height=390)
        self.video_label.pack(pady=10, padx=10)
        
        # Status Row
        status_row = ctk.CTkFrame(self.vision_card, fg_color=CARD)
        status_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.ai_status_badge = ctk.CTkLabel(status_row, text="● HEALTHY", font=("Inter", 12, "bold"),
                                           fg_color="#E6F9F2", text_color=GREEN, corner_radius=8, height=32)
        self.ai_status_badge.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.ai_temp_badge = ctk.CTkLabel(status_row, text="🌡 28.9°C", font=("Inter", 12, "bold"),
                                         fg_color="#E6EEFF", text_color=ACCENT, corner_radius=8, height=32)
        self.ai_temp_badge.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Emergency Button
        self.override_btn = ctk.CTkButton(left_col, text="⊘ EMERGENCY OVERRIDE", 
                                          font=("Inter", 14, "bold"),
                                          fg_color=CARD, text_color=RED,
                                          border_color=RED, border_width=1,
                                          hover_color="#FFF0F3", height=45,
                                          command=self.manual_kill)
        self.override_btn.pack(fill=tk.X, pady=(15, 0))
        
        self.override_sub = ctk.CTkLabel(left_col, text="Triggers immediate motor shutdown", font=("Inter", 10), text_color=SUBTEXT)
        self.override_sub.pack(pady=5)

        # Event Log
        self.log_card = self._create_card(left_col, "SYSTEM EVENT LOG")
        self.log_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.log_box = tk.Text(self.log_card, bg=LOG_BG, fg="white", font=("JetBrains Mono", 10),
                               bd=0, padx=10, pady=10, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # RIGHT PANEL (60%)
        right_col = ctk.CTkFrame(parent, fg_color=BG)
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.grid_columnconfigure(0, weight=1)

        # CNN Fusion Score
        self.fusion_card = self._create_card(right_col, "AI FAULT PROBABILITY")
        self.fusion_card.pack(fill=tk.BOTH, expand=True)
        
        self.fig_f, self.ax_f = plt.subplots(figsize=(6, 5), facecolor=CARD)
        self._style_matplotlib_ax(self.ax_f)
        self.canvas_f = FigureCanvasTkAgg(self.fig_f, master=self.fusion_card)
        self.canvas_f.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ── UI Helpers ───────────────────────────────────────────────
    def _create_card(self, parent, title):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        
        # Section Label (Small Caps style)
        lbl = ctk.CTkLabel(card, text=title, font=("Inter", 10, "bold"), text_color=SUBTEXT)
        lbl.pack(anchor="w", padx=16, pady=(12, 0))
        
        # LIVE Dot Indicator
        live_dot = ctk.CTkLabel(card, text="⟳ LIVE", font=("Inter", 8, "bold"), text_color=ACCENT)
        live_dot.place(relx=0.95, rely=0.04, anchor="ne")
        
        return card

    def _add_spec_row(self, card, label, value):
        row = ctk.CTkFrame(card, fg_color=CARD, height=40)
        row.pack(fill=tk.X, padx=16)
        
        lbl = ctk.CTkLabel(row, text=label, font=("Inter", 10), text_color=SUBTEXT)
        lbl.pack(side=tk.LEFT)
        
        val = ctk.CTkLabel(row, text=value, font=("JetBrains Mono", 10, "bold"), text_color=ACCENT)
        val.pack(side=tk.RIGHT)
        
        sep = ctk.CTkFrame(card, height=1, fg_color=BORDER)
        sep.pack(fill=tk.X, padx=16)

    def _style_matplotlib_ax(self, ax):
        ax.set_facecolor(PANEL)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(BORDER)
        ax.spines['bottom'].set_color(BORDER)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.grid(True, axis='y', color=BORDER, linestyle='--', linewidth=0.5, alpha=0.5)

    # ── Logic & Updates ──────────────────────────────────────────
    def update_video(self, frame):
        if frame is None: return
        try:
            frame = cv2.resize(frame, (520, 390))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except: pass

    def update_graphs(self, waveform, fft_data, current_temp, fusion_score, humidity=0.0, hardware_vib=0.0):
        self._last_temp = current_temp
        self.fusion_scores.append(fusion_score)
        if len(self.fusion_scores) > 60: self.fusion_scores.pop(0)
        
        # Update Main Text Values
        self.temp_val_lbl.configure(text=f"{current_temp:.1f}°C")
        self.hum_lbl.configure(text=f"Humidity: {humidity:.1f}%")
        self.ai_temp_badge.configure(text=f"🌡 {current_temp:.1f}°C")

        # Temperature History
        self._temp_history.append(current_temp)
        if len(self._temp_history) > 60: self._temp_history.pop(0)
        self.ax_t.clear()
        self._style_matplotlib_ax(self.ax_t)
        
        # Color logic based on thresholds
        tc = RED if current_temp >= 35 else YELLOW if current_temp >= 32 else ACCENT
        
        self.ax_t.plot(self._temp_history, color=tc, linewidth=2)
        self.ax_t.fill_between(range(len(self._temp_history)), self._temp_history, color=tc, alpha=0.1)
        
        # Visual threshold lines
        self.ax_t.axhline(32, color=YELLOW, ls='--', lw=1.0, alpha=0.6)
        self.ax_t.axhline(35, color=RED,    ls='--', lw=1.0, alpha=0.6)
        self.canvas_t.draw_idle()

        # Vibration Level (Time Series with Constant Y-Axis)
        if hardware_vib > 0 or len(self._vibe_history) > 0:
            self._vibe_history.append(hardware_vib)
            if len(self._vibe_history) > 60: self._vibe_history.pop(0)
            
            self.ax_v.clear()
            self._style_matplotlib_ax(self.ax_v)
            
            # Vibration logic: >337 is critical (+100), >287 is warning (+50)
            vc = RED if hardware_vib >= 337.0 else YELLOW if hardware_vib >= 287.0 else ACCENT
            
            self.ax_v.plot(self._vibe_history, color=vc, linewidth=2.0)
            self.ax_v.fill_between(range(len(self._vibe_history)), self._vibe_history, color=vc, alpha=0.1)
            
            # Constant Y-axis and dynamic X-axis (scrolling)
            self.ax_v.set_ylim(230, 350)
            self.ax_v.axhline(237, color=SUBTEXT, ls='--', lw=0.8, alpha=0.3) # Baseline
            self.ax_v.axhline(287, color=YELLOW,  ls='--', lw=1.0, alpha=0.5) # Warning Threshold
            self.ax_v.axhline(337, color=RED,     ls='--', lw=1.0, alpha=0.5) # Critical Threshold
            
            self.canvas_v.draw_idle()

        # Acoustic Waveform
        if waveform is not None and len(waveform) > 0:
            self.ax_a.clear()
            self._style_matplotlib_ax(self.ax_a)
            # Style as bar charts for "Equalizer" look
            indices = np.linspace(0, len(waveform)-1, 40).astype(int)
            bars = waveform[indices]
            self.ax_a.bar(range(len(bars)), bars, color=ACCENT, alpha=0.7)
            self.canvas_a.draw_idle()

        # Fusion Score
        self.ax_f.clear()
        self._style_matplotlib_ax(self.ax_f)
        fc = RED if fusion_score > 0.8 else YELLOW if fusion_score > 0.6 else GREEN
        self.ax_f.plot(self.fusion_scores, color=fc, linewidth=2)
        self.ax_f.fill_between(range(len(self.fusion_scores)), self.fusion_scores, color=fc, alpha=0.1)
        self.ax_f.set_ylim(0, 1.1)
        self.canvas_f.draw_idle()

    def update_status(self, status):
        color = GREEN if status == "HEALTHY" else YELLOW if status == "WARNING" else RED
        bg = "#E6F9F2" if status == "HEALTHY" else "#FFF4E5" if status == "WARNING" else "#FFF0F3"
        self.temp_status_pill.configure(text=f"● {status}", text_color=color, fg_color=bg)
        self.ai_status_badge.configure(text=f"● {status}", text_color=color, fg_color=bg)

    def log_event(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_box.see(tk.END)

    def _tick_clock(self):
        self.clock_lbl.configure(text=time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _animate_status_dot(self):
        current = self.status_dot.cget("text_color")
        next_c = BG if current == GREEN else GREEN
        self.status_dot.configure(text_color=next_c)
        self.root.after(800, self._animate_status_dot)

    def manual_kill(self):
        self.log_event(">>> MANUAL OVERRIDE INITIATED! <<<")
        self.update_status("CRITICAL")
        self.fusion_engine.send_kill_command()

    def start(self):
        self.root.mainloop()
