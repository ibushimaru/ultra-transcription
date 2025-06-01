#!/usr/bin/env python3
"""
Ultra Audio Transcription - GUI Interface
Simple drag-and-drop interface for Windows users
"""

import os
import sys
import json
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import queue

class TranscriptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultra Audio Transcription v3.1.0")
        self.root.geometry("800x600")
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        
        # Setup UI
        self.setup_ui()
        
        # Start output monitor
        self.root.after(100, self.check_output_queue)
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Ultra Audio Transcription", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        subtitle_label = ttk.Label(main_frame, text="Powered by Whisper Large-v3 Turbo • 12.6x Faster", 
                                  font=('Arial', 10))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="Audio File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(main_frame, textvariable=self.file_var, width=50)
        self.file_entry.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(main_frame, text="Browse...", 
                  command=self.browse_file).grid(row=2, column=2, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Filler words option
        self.preserve_fillers = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Keep filler words (なるほど、ええ、etc.)", 
                       variable=self.preserve_fillers).grid(row=0, column=0, sticky=tk.W, padx=5)
        
        # Speaker recognition option
        self.enable_speaker = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable speaker recognition", 
                       variable=self.enable_speaker).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Model selection
        ttk.Label(options_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="large-v3-turbo")
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_var, 
                                  values=["large-v3-turbo", "large", "medium", "small"], 
                                  state="readonly", width=15)
        model_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Output log
        log_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        log_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        self.transcribe_btn = ttk.Button(button_frame, text="Start Transcription", 
                                        command=self.start_transcription, style="Accent.TButton")
        self.transcribe_btn.grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, text="Clear Log", 
                  command=self.clear_log).grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).grid(row=0, column=2, padx=5)
        
        # Enable drag and drop
        self.setup_drag_drop()
        
    def setup_drag_drop(self):
        """Enable drag and drop for the file entry"""
        # Simple instruction instead of complex drag-drop
        self.log("Welcome! Select an audio file using Browse button to begin.")
        self.log("Supported formats: MP3, WAV, M4A, FLAC, AAC, OGG, WMA")
            
    def browse_file(self):
        """Open file browser"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.wma"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.file_var.set(filename)
            self.log(f"File selected: {os.path.basename(filename)}")
            
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def clear_log(self):
        """Clear the log window"""
        self.log_text.delete(1.0, tk.END)
        
    def check_output_queue(self):
        """Check for output from background thread"""
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.log(message)
        except queue.Empty:
            pass
        self.root.after(100, self.check_output_queue)
        
    def start_transcription(self):
        """Start transcription in background thread"""
        audio_file = self.file_var.get()
        
        if not audio_file:
            messagebox.showwarning("No File", "Please select an audio file first")
            return
            
        if not os.path.exists(audio_file):
            messagebox.showerror("File Not Found", f"The file does not exist:\n{audio_file}")
            return
            
        # Disable button and start progress
        self.transcribe_btn.config(state='disabled')
        self.progress.start()
        
        # Start transcription in background
        thread = threading.Thread(target=self.run_transcription, args=(audio_file,))
        thread.daemon = True
        thread.start()
        
    def run_transcription(self, audio_file):
        """Run transcription process"""
        try:
            # Build command
            base_name = Path(audio_file).stem
            output_name = f"{base_name}_transcribed"
            
            cmd = [
                sys.executable, "-m", "transcription.rapid_ultra_processor",
                audio_file, "-o", output_name,
                "--model", self.model_var.get()
            ]
            
            if not self.preserve_fillers.get():
                cmd.append("--no-fillers")
                
            if not self.enable_speaker.get():
                cmd.append("--no-speaker")
                
            self.output_queue.put(f"Starting transcription of: {os.path.basename(audio_file)}")
            self.output_queue.put(f"Model: {self.model_var.get()}")
            self.output_queue.put(f"Options: Fillers={'ON' if self.preserve_fillers.get() else 'OFF'}, Speaker={'ON' if self.enable_speaker.get() else 'OFF'}")
            self.output_queue.put("-" * 50)
            
            # Run process
            if os.path.exists("venv/Scripts/python.exe"):
                cmd[0] = "venv/Scripts/python.exe"
                
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output
            for line in process.stdout:
                self.output_queue.put(line.rstrip())
                
            process.wait()
            
            if process.returncode == 0:
                self.output_queue.put("-" * 50)
                self.output_queue.put("✅ Transcription completed successfully!")
                self.output_queue.put(f"Output files:")
                self.output_queue.put(f"  • {output_name}_ultra_precision.txt")
                self.output_queue.put(f"  • {output_name}_ultra_precision.json")
                self.output_queue.put(f"  • {output_name}_ultra_precision.csv")
                self.output_queue.put(f"  • {output_name}_ultra_precision.srt")
                
                # Ask to open output folder
                self.root.after(0, lambda: self.ask_open_folder(os.path.dirname(os.path.abspath(audio_file))))
            else:
                self.output_queue.put(f"❌ Error: Process exited with code {process.returncode}")
                
        except Exception as e:
            self.output_queue.put(f"❌ Error: {str(e)}")
            
        finally:
            # Re-enable button and stop progress
            self.root.after(0, lambda: self.transcribe_btn.config(state='normal'))
            self.root.after(0, self.progress.stop)
            
    def ask_open_folder(self, folder_path):
        """Ask user if they want to open the output folder"""
        if messagebox.askyesno("Transcription Complete", 
                              "Transcription completed successfully!\n\nDo you want to open the output folder?"):
            os.startfile(folder_path)

def main():
    # Check if we're in the right directory
    if not os.path.exists("transcription"):
        messagebox.showerror("Error", "Please run this from the Ultra Audio Transcription directory")
        sys.exit(1)
        
    root = tk.Tk()
    
    # Try to use modern theme
    try:
        root.tk.call('source', 'azure.tcl')
        root.tk.call('set_theme', 'dark')
    except:
        try:
            style = ttk.Style()
            style.theme_use('clam')
        except:
            pass
    
    app = TranscriptionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()