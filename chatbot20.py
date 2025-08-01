import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import time
from datetime import datetime
import os
import json
import numpy as np
try:
    import cv2
    from PIL import Image, ImageTk
    import mediapipe as mp
    CAMERA_AVAILABLE = True
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    CAMERA_AVAILABLE = False
    MEDIAPIPE_AVAILABLE = False
    missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Missing dependency: {missing_lib}")
    print("Please install required packages:")
    print("pip install opencv-python pillow mediapipe")
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Text-to-speech not available. Install pyttsx3:")
    print("pip install pyttsx3")

class Vellora:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Assistant")
        self.root.geometry("900x750")
        self.root.minsize(800, 600)
        self.is_video_mode = False
        self.is_mic_on = False
        self.is_audio_enabled = False
        self.is_sign_language_mode = False
        self.video_capture = None
        self.camera_running = False
        self.sign_detection_running = False
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        self.mp_hands = mp.solutions.hands if MEDIAPIPE_AVAILABLE else None
        self.mp_drawing = mp.solutions.drawing_utils if MEDIAPIPE_AVAILABLE else None
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                      max_num_hands=2,
                                      min_detection_confidence=0.5) if MEDIAPIPE_AVAILABLE else None
        self.sign_dict = {
            "THUMBS_UP": "Yes/Good",
            "THUMBS_DOWN": "No/Bad",
            "OPEN_PALM": "Hello/Stop",
            "CLOSED_FIST": "Wait",
            "PEACE_SIGN": "Peace/Two",
            "POINTING_UP": "Attention/Question",
            "PINCH": "Small/Little"
        }
        self.sign_responses = self._load_sign_responses()
        self.current_response_sign = None
        self.messages = [
            {"id": 1, "text": "Hello! I can detect sign language and respond with signs. How can I help you today?", 
             "sender": "bot", "timestamp": self._get_timestamp(), "sign_key": "OPEN_PALM"}
        ]
        self._create_ui()
        self._display_messages()
        if not CAMERA_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            self._show_message("Please install 'opencv-python', 'pillow', and 'mediapipe' packages for sign language features.")
        if not TTS_AVAILABLE:
            self._show_message("Please install 'pyttsx3' package for text-to-speech features.")

    def _load_sign_responses(self):
        responses = {}
        for sign in self.sign_dict.keys():
            responses[sign] = f"[Sign for: {self.sign_dict[sign]}]"         
        return responses 

    def _create_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.left_frame = ttk.Frame(self.main_paned, style="LeftFrame.TFrame")
        self.main_paned.add(self.left_frame, weight=1)
        self.right_frame = ttk.Frame(self.main_paned, style="RightFrame.TFrame")
        self.main_paned.add(self.right_frame, weight=1)
        self._create_video_area()
        self._create_chat_area()
        self._create_chat_area()
        self._apply_styles()    

    def _create_video_area(self):
        video_header = ttk.Frame(self.left_frame, style="Header.TFrame")
        video_header.pack(fill=tk.X, pady=(0, 5))    
        title_label = ttk.Label(video_header, text="Vellora", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        controls_frame = ttk.Frame(video_header)
        controls_frame.pack(side=tk.RIGHT, padx=5)    
        self.sign_button = ttk.Button(controls_frame, text="👋 Sign Mode", command=self._toggle_sign_language)
        self.sign_button.pack(side=tk.RIGHT, padx=5)    
        self.camera_button = ttk.Button(controls_frame, text="📹 Camera", command=self._toggle_video)
        self.camera_button.pack(side=tk.RIGHT, padx=5)
        self.video_frame = ttk.Frame(self.left_frame, relief="sunken", borderwidth=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_canvas = tk.Canvas(self.video_frame, bg="red")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.camera_status_label = ttk.Label(self.video_frame, 
                                           text="Camera Off - Click '📹 Camera' to start",
                                           background="black", foreground="white")
        self.camera_status_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.detection_frame = ttk.Frame(self.left_frame)
        self.detection_frame.pack(fill=tk.X, pady=5)  
        self.detection_label = ttk.Label(self.detection_frame, text="Detection: Inactive", font=("Arial", 10))
        self.detection_label.pack(side=tk.LEFT, padx=10)     
        self.current_sign_label = ttk.Label(self.detection_frame, text="Detected Sign: None", font=("Arial", 10, "bold"))
        self.current_sign_label.pack(side=tk.RIGHT, padx=10)
        ref_frame = ttk.LabelFrame(self.left_frame, text="Sign Language Guide")
        ref_frame.pack(fill=tk.X, padx=5, pady=5)    
        guide_text = "• 👍 - Yes/Good\n• 👎 - No/Bad\n• ✋ - Hello/Stop\n• ✊ - Wait\n• ✌ - Peace/Two\n• ☝ - Question\n• 👌 - Small/Little"
        guide_label = ttk.Label(ref_frame, text=guide_text, justify=tk.LEFT)
        guide_label.pack(padx=10, pady=5, anchor=tk.W)
        self.sign_visual_frame = ttk.LabelFrame(self.left_frame, text="Bot Sign Response")
        self.sign_visual_frame.pack(fill=tk.X, padx=5, pady=5)     
        self.sign_visual_canvas = tk.Canvas(self.sign_visual_frame, height=150, bg="yellow")
        self.sign_visual_canvas.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self._draw_default_hand()    

    def _draw_default_hand(self):
        self.sign_visual_canvas.delete("all")
        width = self.sign_visual_canvas.winfo_width() or 300
        height = self.sign_visual_canvas.winfo_height() or 150
        cx, cy = width/2, height/2
        self.sign_visual_canvas.create_oval(cx-30, cy-20, cx+30, cy+40, outline="gray", width=2)
        self.sign_visual_canvas.create_line(cx-30, cy, cx-50, cy-30, smooth=True, width=2, fill="gray")
        self.sign_visual_canvas.create_line(cx-15, cy-20, cx-15, cy-70, smooth=True, width=2, fill="gray")
        self.sign_visual_canvas.create_line(cx, cy-20, cx, cy-80, smooth=True, width=2, fill="gray")
        self.sign_visual_canvas.create_line(cx+15, cy-20, cx+15, cy-70, smooth=True, width=2, fill="gray") 
        self.sign_visual_canvas.create_line(cx+30, cy-20, cx+30, cy-60, smooth=True, width=2, fill="gray")
        self.sign_visual_canvas.create_text(cx, cy+70, text="Ready for sign response", fill="gray")    

    def _draw_sign(self, sign_key):
        self.sign_visual_canvas.delete("all")    
        width = self.sign_visual_canvas.winfo_width() or 300
        height = self.sign_visual_canvas.winfo_height() or 150
        cx, cy = width/2, height/2    
        if sign_key == "THUMBS_UP":
            self.sign_visual_canvas.create_oval(cx-20, cy-20, cx+20, cy+20, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_rectangle(cx-15, cy, cx+15, cy+60, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_line(cx-15, cy+60, cx+15, cy+60, width=2)
            self.sign_visual_canvas.create_text(cx, cy+80, text="THUMBS UP - Yes/Good", font=("Arial", 12, "bold"))       
        elif sign_key == "THUMBS_DOWN":
            self.sign_visual_canvas.create_oval(cx-20, cy-20, cx+20, cy+20, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_rectangle(cx-15, cy-60, cx+15, cy, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_line(cx-15, cy-60, cx+15, cy-60, width=2)
            self.sign_visual_canvas.create_text(cx, cy+40, text="THUMBS DOWN - No/Bad", font=("Arial", 12, "bold"))       
        elif sign_key == "OPEN_PALM":
            self.sign_visual_canvas.create_oval(cx-30, cy+30, cx+30, cy+70, fill="#4a6fa5", outline="black")
            for i in range(-20, 30, 10):
                self.sign_visual_canvas.create_rectangle(cx+i-5, cy-50, cx+i+5, cy+30, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_text(cx, cy+90, text="OPEN PALM - Hello/Stop", font=("Arial", 12, "bold"))       
        elif sign_key == "CLOSED_FIST":
            self.sign_visual_canvas.create_oval(cx-30, cy-30, cx+30, cy+30, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_text(cx, cy+50, text="CLOSED FIST - Wait", font=("Arial", 12, "bold"))       
        elif sign_key == "PEACE_SIGN":
            self.sign_visual_canvas.create_oval(cx-25, cy+20, cx+25, cy+60, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_rectangle(cx-20, cy, cx-5, cy-50, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_rectangle(cx+5, cy, cx+20, cy-50, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_text(cx, cy+80, text="PEACE SIGN - Peace/Two", font=("Arial", 12, "bold"))       
        elif sign_key == "POINTING_UP":
            self.sign_visual_canvas.create_oval(cx-25, cy+30, cx+25, cy+70, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_rectangle(cx-5, cy-60, cx+5, cy+30, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_text(cx, cy+90, text="POINTING UP - Question/Attention", font=("Arial", 12, "bold"))        
        elif sign_key == "PINCH":
            self.sign_visual_canvas.create_oval(cx-25, cy+20, cx+25, cy+60, fill="#4a6fa5", outline="black")
            self.sign_visual_canvas.create_arc(cx-20, cy-30, cx+20, cy+10, start=0, extent=180, outline="black", width=2)
            self.sign_visual_canvas.create_text(cx, cy+80, text="PINCH - Small/Little", font=("Arial", 12, "bold"))        
        else:
            self._draw_default_hand()

    def _create_chat_area(self):
        chat_header = ttk.Frame(self.right_frame, style="Header.TFrame")
        chat_header.pack(fill=tk.X, pady=(0, 5))    
        title_label = ttk.Label(chat_header, text="Chat", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.settings_button = ttk.Button(chat_header, text="⚙", width=3, command=self._show_settings)
        self.settings_button.pack(side=tk.RIGHT, padx=5, pady=5)    
        self.audio_button = ttk.Button(chat_header, text="🔇", width=3, command=self._toggle_audio)
        self.audio_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.chat_frame = ttk.Frame(self.right_frame)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)   
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, width=40, height=15, state="disabled")
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.tag_configure("bot", background="#e9e9e9", lmargin1=10, lmargin2=10, rmargin=60)
        self.chat_display.tag_configure("user", background="#2c75d8", foreground="white", rmargin=10, lmargin1=60, lmargin2=60)
        self.chat_display.tag_configure("sign", background="#FFD700", foreground="black", lmargin1=10, lmargin2=10, rmargin=60)
        self.sign_response_frame = ttk.LabelFrame(self.right_frame, text="Sign Language Response")
        self.sign_response_frame.pack(fill=tk.X, padx=5, pady=5)    
        self.sign_response_label = ttk.Label(self.sign_response_frame, text="No response yet", font=("Arial", 12))
        self.sign_response_label.pack(padx=10, pady=10)
        self._create_chat_input_area()     

    def _create_chat_input_area(self):
        self.input_frame = ttk.Frame(self.right_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        self.text_input = ttk.Entry(self.input_frame)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.text_input.bind("<Return>", lambda event: self._send_message())
        self.mic_button = ttk.Button(self.input_frame, text="🎤", width=3, command=self._toggle_mic)
        self.mic_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.send_button = ttk.Button(self.input_frame, text="Send", command=self._send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.quick_responses_frame = ttk.Frame(self.right_frame)
        self.quick_responses_frame.pack(fill=tk.X, padx=5, pady=(0, 5))   
        quick_responses = ["Hello", "Thank you", "Yes", "No", "Help", "I need assistance"]    
        for i, response in enumerate(quick_responses):
            btn = ttk.Button(self.quick_responses_frame, text=response, 
                           command=lambda r=response: self._use_quick_response(r))
            row = i // 3
            col = i % 3
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        for i in range(3):
            self.quick_responses_frame.columnconfigure(i, weight=1)

    def _apply_styles(self):
        style = ttk.Style()
        style.configure("Header.TFrame", background="#4a6fa5")
        style.configure("Header.TLabel", font=("Arial", 16, "bold"), foreground="white", background="#4a6fa5")
        style.configure("TButton", font=("Arial", 10))
        style.configure("LeftFrame.TFrame", background="#f0f0f0")
        style.configure("RightFrame.TFrame", background="#ffffff")

    def _display_messages(self):
        self.chat_display.config(state="normal")
        self.chat_display.delete(1.0, tk.END)    
        for msg in self.messages:
            tag = msg["sender"]
            prefix = f"[{msg['timestamp']}] "
            if tag == "bot":
                prefix += "Bot: "
            elif tag == "sign":
                prefix += "Sign Detected: "
            else:
                prefix += "You: "            
            message_with_prefix = prefix + msg["text"] + "\n\n"
            self.chat_display.insert(tk.END, message_with_prefix, tag)
            if "sign_key" in msg and msg["sender"] == "bot":
                self._update_sign_response(msg["sign_key"])        
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

    def _update_sign_response(self, sign_key):
        if sign_key in self.sign_responses:
            self.sign_response_label.config(text=self.sign_responses[sign_key])
            self._draw_sign(sign_key)
            self.current_response_sign = sign_key
        else:
            self.sign_response_label.config(text="No sign response available")
            self._draw_default_hand()
            self.current_response_sign = None          

    def _send_message(self):
        message_text = self.text_input.get().strip()
        if not message_text:
            return
        self.messages.append({
            "id": len(self.messages) + 1,
            "text": message_text,
            "sender": "user",
            "timestamp": self._get_timestamp()
        })
        self.text_input.delete(0, tk.END)
        self._display_messages()
        sign_key = self._text_to_sign(message_text)
        self.root.after(500, lambda: self._send_bot_response(message_text, sign_key))

    def _text_to_sign(self, text):
        text = text.lower()
        if any(word in text for word in ["hello", "hi", "hey", "greet"]):
            return "OPEN_PALM"
        elif any(word in text for word in ["yes", "good", "ok", "okay", "fine", "agree"]):
            return "THUMBS_UP"
        elif any(word in text for word in ["no", "bad", "wrong", "disagree", "don't"]):
            return "THUMBS_DOWN"
        elif any(word in text for word in ["wait", "hold", "stop", "pause"]):
            return "CLOSED_FIST"
        elif any(word in text for word in ["peace", "two", "2", "both"]):
            return "PEACE_SIGN"
        elif any(word in text for word in ["question", "what", "how", "when", "where", "why", "?"]):
            return "POINTING_UP"
        elif any(word in text for word in ["small", "little", "tiny", "bit"]):
            return "PINCH"
        else:
            return "OPEN_PALM" 

    def _send_bot_response(self, user_message, sign_key):
        if "hello" in user_message.lower() or "hi" in user_message.lower():
            response_text = "Hello! How can I help you today?"
        elif "thank" in user_message.lower():
            response_text = "You're welcome! Is there anything else you need?"
            sign_key = "THUMBS_UP"
        elif "help" in user_message.lower():
            response_text = "I'm here to help. You can use sign language or type your questions."
            sign_key = "OPEN_PALM"
        elif any(word in user_message.lower() for word in ["yes", "yeah", "ok", "sure"]):
            response_text = "Great! What would you like to do next?"
            sign_key = "THUMBS_UP"
        elif any(word in user_message.lower() for word in ["no", "nope", "don't", "not"]):
            response_text = "I understand. Let me know if you need something else."
            sign_key = "THUMBS_DOWN"
        else:
            response_text = "I understand your message. How else can I assist you?"
        self.messages.append({
            "id": len(self.messages) + 1,
            "text": response_text,
            "sender": "bot",
            "timestamp": self._get_timestamp(),
            "sign_key": sign_key
        })   
        self._display_messages()
        if self.is_audio_enabled and TTS_AVAILABLE:
            self._speak_text(response_text)    

    def _speak_text(self, text):
        if not TTS_AVAILABLE:
            return
        threading.Thread(target=lambda: self.tts_engine.say(text) or self.tts_engine.runAndWait(), 
                       daemon=True).start()   

    def _use_quick_response(self, response):
        self.text_input.delete(0, tk.END)
        self.text_input.insert(0, response)
        self._send_message()    

    def _toggle_video(self):
        if not CAMERA_AVAILABLE:
            self._show_message("Camera functionality requires OpenCV and PIL. Please install these libraries.")
            return        
        if self.is_video_mode:
            self.is_video_mode = False
            self.camera_button.config(text="📹 Camera")
            self.camera_status_label.config(text="Camera Off - Click '📹 Camera' to start")
            self.camera_status_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            if self.camera_running:
                self.camera_running = False
                if self.video_capture:
                    self.video_capture.release()
            if self.is_sign_language_mode:
                self._toggle_sign_language()
        else:
            self.is_video_mode = True
            self.camera_button.config(text="📹 Camera ✓")
            self._start_camera()         

    def _start_camera(self):
        if not CAMERA_AVAILABLE:
            self.camera_status_label.config(text="Camera not available. Install OpenCV and PIL.")
            return
        self.camera_running = True
        threading.Thread(target=self._camera_thread, daemon=True).start()

    def _camera_thread(self):
        if not CAMERA_AVAILABLE:
            return 
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self.root.after(0, lambda: self.camera_status_label.config(
                    text="Could not access camera. Check permissions."
                ))
                self.camera_running = False
                return
            self.root.after(0, lambda: self.camera_status_label.place_forget()) 
            while self.camera_running:
                ret, frame = self.video_capture.read()
                if ret:
                    if self.is_sign_language_mode and MEDIAPIPE_AVAILABLE:
                        frame = self._process_sign_language(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (640, 480))
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.root.after(0, lambda f=imgtk: self._update_video_canvas(f))
                time.sleep(0.03)
            if self.video_capture:
                self.video_capture.release()
        except Exception as e:
            self.root.after(0, lambda: self._show_message(f"Camera error: {str(e)}"))
            self.camera_running = False

    def _process_sign_language(self, frame):
        if not MEDIAPIPE_AVAILABLE:
            return frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                detected_sign = self._recognize_hand_gesture(hand_landmarks)            
                if detected_sign:
                    self.root.after(0, lambda sign=detected_sign: self._update_sign_detection(sign))
                    if (not self.messages or 
                        self.messages[-1]["sender"] != "sign" or
                        self.messages[-1]["text"] != self.sign_dict.get(detected_sign, "Unknown sign")):                   
                        self.messages.append({
                            "id": len(self.messages) + 1,
                            "text": self.sign_dict.get(detected_sign, "Unknown sign"),
                            "sender": "sign",
                            "timestamp": self._get_timestamp(),
                            "sign_key": detected_sign
                        })
                        self.root.after(0, self._display_messages)
                        self.root.after(3000, lambda s=detected_sign: self._respond_to_sign(s))     
        return frame  

    def _recognize_hand_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        if (thumb_tip.y < thumb_ip.y and 
            index_tip.y > index_pip.y and 
            middle_tip.y > middle_pip.y):
            return "THUMBS_UP"
        elif (thumb_tip.y > thumb_ip.y and 
              index_tip.y > index_pip.y and 
              middle_tip.y > middle_pip.y):
            return "THUMBS_DOWN"
        elif (index_tip.y < index_pip.y and 
              middle_tip.y < middle_pip.y and 
              ring_tip.y < ring_pip.y and 
              pinky_tip.y < pinky_pip.y):
            return "OPEN_PALM"
        elif (index_tip.y > index_pip.y and 
              middle_tip.y > middle_pip.y and 
              ring_tip.y > ring_pip.y and 
              pinky_tip.y > pinky_pip.y):
            return "CLOSED_FIST"
        elif (index_tip.y < index_pip.y and 
              middle_tip.y < middle_tip.y and 
              ring_tip.y > ring_pip.y and 
              pinky_tip.y > pinky_pip.y):
            return "PEACE_SIGN"
        elif (index_tip.y < index_pip.y and 
              middle_tip.y > middle_pip.y and 
              ring_tip.y > ring_pip.y and 
              pinky_tip.y > pinky_pip.y):
            return "POINTING_UP"      
        elif (abs(thumb_tip.x - index_tip.x) < 0.05 and 
              abs(thumb_tip.y - index_tip.y) < 0.05):
            return "PINCH"
        return None   

    def _respond_to_sign(self, sign_key):
        if sign_key == "THUMBS_UP":
            response = "Great! I'm glad you agree."
        elif sign_key == "THUMBS_DOWN":
            response = "I understand you disagree or need something different."
        elif sign_key == "OPEN_PALM":
            response = "Hello! How can I help you?"
        elif sign_key == "CLOSED_FIST":
            response = "I'll wait for your next instruction."
        elif sign_key == "PEACE_SIGN":
            response = "Peace! Do you need help with two options?"
        elif sign_key == "POINTING_UP":
            response = "Do you have a question? I'm listening."
        elif sign_key == "PINCH":
            response = "I understand you're referring to something small or precise."
        else:
            response = "I see you're using sign language. How can I help?"
        self.messages.append({
            "id": len(self.messages) + 1,
            "text": response,
            "sender": "bot",
            "timestamp": self._get_timestamp(),
            "sign_key": sign_key
        })
        self._display_messages()     
        if self.is_audio_enabled and TTS_AVAILABLE:
            self._speak_text(response)  

    def _update_video_canvas(self, imgtk):
        self.video_canvas.imgtk = imgtk 
        self.video_canvas.config(width=imgtk.width(), height=imgtk.height())
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)  

    def _update_sign_detection(self, sign):
        if sign in self.sign_dict:
            self.current_sign_label.config(text=f"Detected Sign: {self.sign_dict[sign]}")
        else:
            self.current_sign_label.config(text="Detected Sign: Unknown")    
        self.detection_label.config(text="Detection: Active") 

    def _toggle_sign_language(self):
        if not MEDIAPIPE_AVAILABLE:
            self._show_message("Vellora requires MediaPipe. Please install this library.")
            return           
        if not self.is_video_mode:
            self._toggle_video()           
        if self.is_sign_language_mode:
            self.is_sign_language_mode = False
            self.sign_button.config(text="👋 Sign Mode")
            self.detection_label.config(text="Detection: Inactive")
            self.current_sign_label.config(text="Detected Sign: None")
        else:
            self.is_sign_language_mode = True
            self.sign_button.config(text="👋 Sign Mode ✓")
            self.detection_label.config(text="Detection: Active")   

    def _toggle_audio(self):
        if not TTS_AVAILABLE:
            self._show_message("Text-to-speech requires pyttsx3. Please install this library.")
            return           
        if self.is_audio_enabled:
            self.is_audio_enabled = False
            self.audio_button.config(text="🔇")
        else:
            self.is_audio_enabled = True
            self.audio_button.config(text="🔊")           
            self._speak_text("Audio enabled")   

    def _toggle_mic(self):
        self._show_message("Speech-to-text functionality not implemented in this version.")   

    def _show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        settings_window.grab_set()       
        ttk.Label(settings_window, text="Settings", font=("Arial", 14, "bold")).pack(pady=10)       
        camera_frame = ttk.LabelFrame(settings_window, text="Camera Settings")
        camera_frame.pack(fill=tk.X, padx=10, pady=5)       
        ttk.Button(camera_frame, text="Test Camera", 
                 command=lambda: self._show_message("Camera test not implemented")).pack(padx=10, pady=5)    
        audio_frame = ttk.LabelFrame(settings_window, text="Audio Settings")
        audio_frame.pack(fill=tk.X, padx=10, pady=5)    
        ttk.Button(audio_frame, text="Test Audio", 
                 command=lambda: self._speak_text("This is a test of the text to speech system") 
                 if TTS_AVAILABLE else self._show_message("Text-to-speech not available")).pack(padx=10, pady=5)    
        sign_frame = ttk.LabelFrame(settings_window, text="Sign Language Settings")
        sign_frame.pack(fill=tk.X, padx=10, pady=5)    
        ttk.Button(sign_frame, text="Show Sign Guide", 
                 command=self._show_sign_guide).pack(padx=10, pady=5)   
        about_frame = ttk.LabelFrame(settings_window, text="About")
        about_frame.pack(fill=tk.X, padx=10, pady=5)    
        ttk.Label(about_frame, text="Sign Language Assistant v1.0").pack(padx=10, pady=5)
        ttk.Label(about_frame, text="© 2025 Your Name").pack(padx=10, pady=0)  
        ttk.Button(settings_window, text="Close", command=settings_window.destroy).pack(pady=10)

    def _show_sign_guide(self):
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Sign Language Guide")
        guide_window.geometry("500x400")
        guide_window.transient(self.root)    
        ttk.Label(guide_window, text="Supported Sign Language Gestures", 
                font=("Arial", 14, "bold")).pack(pady=10)    
        for sign_key, sign_meaning in self.sign_dict.items():
            sign_frame = ttk.Frame(guide_window)
            sign_frame.pack(fill=tk.X, padx=10, pady=5)       
            sign_canvas = tk.Canvas(sign_frame, width=100, height=100, bg="white")
            sign_canvas.pack(side=tk.LEFT, padx=10, pady=5)        
            self._draw_simple_sign(sign_canvas, sign_key)      
            description_frame = ttk.Frame(sign_frame)
            description_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)       
            ttk.Label(description_frame, text=sign_meaning, font=("Arial", 12, "bold")).pack(anchor=tk.W)
            ttk.Label(description_frame, text=self._get_sign_description(sign_key)).pack(anchor=tk.W)  
        ttk.Button(guide_window, text="Close", command=guide_window.destroy).pack(pady=10)

    def _draw_simple_sign(self, canvas, sign_key):
        width, height = 100, 100
        cx, cy = width / 2, height / 2     
        if sign_key == "THUMBS_UP":
            canvas.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, fill="#4a6fa5", outline="black")
            canvas.create_rectangle(cx - 10, cy, cx + 10, cy + 40, fill="#4a6fa5", outline="black")
        elif sign_key == "THUMBS_DOWN":
            canvas.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, fill="#4a6fa5", outline="black")
            canvas.create_rectangle(cx - 10, cy - 40, cx + 10, cy, fill="#4a6fa5", outline="black")
        elif sign_key == "OPEN_PALM":
            canvas.create_oval(cx - 20, cy + 20, cx + 20, cy + 40, fill="#4a6fa5", outline="black")
            for i in range(-15, 20, 8):
                canvas.create_rectangle(cx + i - 3, cy - 30, cx + i + 3, cy + 20, fill="#4a6fa5", outline="black")
        elif sign_key == "CLOSED_FIST":
            canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill="#4a6fa5", outline="black")
        elif sign_key == "PEACE_SIGN":
            canvas.create_oval(cx - 15, cy + 10, cx + 15, cy + 40, fill="#4a6fa5", outline="black")
            canvas.create_rectangle(cx - 10, cy, cx - 3, cy - 30, fill="#4a6fa5", outline="black")
            canvas.create_rectangle(cx + 3, cy, cx + 10, cy - 30, fill="#4a6fa5", outline="black")
        elif sign_key == "POINTING_UP":
            canvas.create_oval(cx - 15, cy + 20, cx + 15, cy + 40, fill="#4a6fa5", outline="black")
            canvas.create_rectangle(cx - 3, cy - 40, cx + 3, cy + 20, fill="#4a6fa5", outline="black")
        elif sign_key == "PINCH":
            canvas.create_oval(cx - 15, cy + 10, cx + 15, cy + 30, fill="#4a6fa5", outline="black")
            canvas.create_arc(cx - 10, cy - 20, cx + 10, cy + 5, start=0, extent=180, outline="black", width=2)

    def _get_sign_description(self, sign_key):
        descriptions = {
            "THUMBS_UP": "Extend thumb upward with other fingers closed to show approval or agreement.",
            "THUMBS_DOWN": "Extend thumb downward with other fingers closed to show disapproval or disagreement.",
            "OPEN_PALM": "Extend all fingers with palm facing forward to say hello or indicate stop.",
            "CLOSED_FIST": "Close all fingers to form a fist to indicate wait or hold.",
            "PEACE_SIGN": "Extend index and middle fingers while keeping other fingers closed to indicate peace or the number two.",
            "POINTING_UP": "Extend only the index finger upward to indicate attention or to ask a question.",
            "PINCH": "Touch the tips of thumb and index finger to indicate something small or precise."
        }   
        return descriptions.get(sign_key, "No description available")    
    def _show_message(self, message):
        messagebox = tk.Toplevel(self.root)
        messagebox.title("Message")
        messagebox.geometry("300x150")
        messagebox.transient(self.root)
        messagebox.grab_set()     
        ttk.Label(messagebox, text=message, wraplength=280).pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        ttk.Button(messagebox, text="OK", command=messagebox.destroy).pack(pady=10)
        messagebox.after(3000, messagebox.destroy) 
    def _get_timestamp(self):
        return datetime.now().strftime("%H:%M") 

    def run(self):
        self.root.mainloop()     

    def cleanup(self):
        if self.camera_running:
            self.camera_running = False      
        if self.video_capture:
            self.video_capture.release()      
        if TTS_AVAILABLE:
            self.tts_engine.stop()

if __name__ == "__main__":
    root = tk.Tk()
    app = Vellora(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    app.run()