#!/usr/bin/env python3
"""
Ultra Audio Transcription v3.2.1
One-file solution for Windows users
"""

import os
import sys
import subprocess
import platform
import time
import json
import argparse
from pathlib import Path

# Check if we're running as compiled exe
IS_EXE = getattr(sys, 'frozen', False)

class UltraTranscribe:
    def __init__(self):
        self.version = "3.2.1"
        self.venv_path = "venv"
        self.setup_marker = ".setup_complete"
        
    def print_header(self):
        """Print application header"""
        print("=" * 60)
        print(f"Ultra Audio Transcription v{self.version}")
        print("Powered by Whisper Large-v3 Turbo - 12.6x Faster")
        print("=" * 60)
        print()
    
    def check_setup(self):
        """Check if setup is complete"""
        return os.path.exists(self.setup_marker) and os.path.exists(self.venv_path)
    
    def get_python_exe(self):
        """Get python executable path"""
        if platform.system() == "Windows":
            return os.path.join(self.venv_path, "Scripts", "python.exe")
        return os.path.join(self.venv_path, "bin", "python")
    
    def get_pip_exe(self):
        """Get pip executable path"""
        if platform.system() == "Windows":
            return os.path.join(self.venv_path, "Scripts", "pip.exe")
        return os.path.join(self.venv_path, "bin", "pip")
    
    def run_setup(self):
        """Run initial setup"""
        print("First time setup detected...")
        print("This will take 5-10 minutes.\n")
        
        # Check Python
        print("Checking Python version...")
        if sys.version_info < (3, 8):
            print("ERROR: Python 3.8+ required")
            print("Please download from: https://python.org")
            input("\nPress Enter to exit...")
            return False
        
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected\n")
        
        # Create virtual environment
        if not os.path.exists(self.venv_path):
            print("Creating virtual environment...")
            try:
                subprocess.run([sys.executable, "-m", "venv", self.venv_path], check=True)
                print("✓ Virtual environment created\n")
            except Exception as e:
                print(f"ERROR creating virtual environment: {e}")
                return False
        
        # Install packages
        print("Installing packages...")
        pip_exe = self.get_pip_exe()
        
        packages = [
            # Core packages
            ("pip", None),
            ("wheel setuptools", None),
            # PyTorch with CUDA
            ("torch torchvision torchaudio", "--index-url https://download.pytorch.org/whl/cu121"),
            # Transcription packages
            ("openai-whisper", None),
            ("librosa", None),
            ("soundfile", None),
            ("pyannote.audio", None),
            # Utilities
            ("tqdm rich", None),
        ]
        
        for package_set, extra_args in packages:
            print(f"  Installing {package_set.split()[0]}...")
            cmd = [pip_exe, "install", "--upgrade"] + package_set.split()
            if extra_args:
                cmd.extend(extra_args.split())
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print(f"  ✓ {package_set.split()[0]} installed")
            except:
                print(f"  ⚠ {package_set.split()[0]} installation failed (will retry on first use)")
        
        # Download model
        print("\nDownloading Whisper Turbo model (2GB)...")
        print("This happens only once...")
        
        # Mark setup complete
        with open(self.setup_marker, "w") as f:
            f.write(f"Setup completed on {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n✅ Setup complete!\n")
        return True
    
    def run_gui(self):
        """Run GUI interface"""
        try:
            # Try to import tkinter
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
            import threading
            import queue
            
            class TranscribeGUI:
                def __init__(self, root, app):
                    self.root = root
                    self.app = app
                    self.root.title(f"Ultra Audio Transcription v{app.version}")
                    self.root.geometry("700x500")
                    
                    self.setup_ui()
                    
                def setup_ui(self):
                    # Main frame
                    main_frame = ttk.Frame(self.root, padding="10")
                    main_frame.pack(fill=tk.BOTH, expand=True)
                    
                    # Title
                    title = ttk.Label(main_frame, text="Ultra Audio Transcription", 
                                     font=('Arial', 16, 'bold'))
                    title.pack(pady=10)
                    
                    # File selection
                    file_frame = ttk.Frame(main_frame)
                    file_frame.pack(fill=tk.X, pady=10)
                    
                    ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT, padx=5)
                    self.file_var = tk.StringVar()
                    self.file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=50)
                    self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                    
                    ttk.Button(file_frame, text="Browse", 
                              command=self.browse_file).pack(side=tk.LEFT, padx=5)
                    
                    # Options
                    options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
                    options_frame.pack(fill=tk.X, pady=10)
                    
                    self.preserve_fillers = tk.BooleanVar(value=True)
                    ttk.Checkbutton(options_frame, text="Keep filler words (なるほど、ええ等)", 
                                   variable=self.preserve_fillers).pack(anchor=tk.W)
                    
                    self.enable_speaker = tk.BooleanVar(value=True)
                    ttk.Checkbutton(options_frame, text="Enable speaker recognition", 
                                   variable=self.enable_speaker).pack(anchor=tk.W)
                    
                    # Output
                    output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
                    output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
                    
                    self.output_text = tk.Text(output_frame, height=10, wrap=tk.WORD)
                    scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
                    self.output_text.configure(yscrollcommand=scrollbar.set)
                    
                    self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    # Buttons
                    button_frame = ttk.Frame(main_frame)
                    button_frame.pack(pady=10)
                    
                    self.transcribe_btn = ttk.Button(button_frame, text="Start Transcription", 
                                                    command=self.start_transcription)
                    self.transcribe_btn.pack(side=tk.LEFT, padx=5)
                    
                    ttk.Button(button_frame, text="Exit", 
                              command=self.root.quit).pack(side=tk.LEFT, padx=5)
                    
                    self.log("Ready. Select an audio file to begin.")
                    
                def browse_file(self):
                    filename = filedialog.askopenfilename(
                        title="Select Audio File",
                        filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac"), 
                                  ("All Files", "*.*")]
                    )
                    if filename:
                        self.file_var.set(filename)
                        self.log(f"Selected: {os.path.basename(filename)}")
                
                def log(self, message):
                    self.output_text.insert(tk.END, f"{message}\\n")
                    self.output_text.see(tk.END)
                    self.root.update()
                
                def start_transcription(self):
                    audio_file = self.file_var.get()
                    if not audio_file:
                        messagebox.showwarning("No File", "Please select an audio file")
                        return
                    
                    self.transcribe_btn.config(state='disabled')
                    self.log("\\nStarting transcription...")
                    
                    # Run in thread
                    thread = threading.Thread(target=self.run_transcription, args=(audio_file,))
                    thread.daemon = True
                    thread.start()
                
                def run_transcription(self, audio_file):
                    try:
                        output_name = Path(audio_file).stem + "_transcribed"
                        
                        cmd = [self.app.get_python_exe(), "-m", "transcription.rapid_ultra_processor",
                               audio_file, "-o", output_name]
                        
                        if not self.preserve_fillers.get():
                            cmd.append("--no-fillers")
                        if not self.enable_speaker.get():
                            cmd.append("--no-speaker")
                        
                        # Run process
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                                 stderr=subprocess.STDOUT, text=True)
                        
                        for line in process.stdout:
                            self.log(line.rstrip())
                        
                        process.wait()
                        
                        if process.returncode == 0:
                            self.log("\\n✅ Transcription complete!")
                            self.log(f"Output saved as: {output_name}_ultra_precision.*")
                        else:
                            self.log("\\n❌ Transcription failed")
                            
                    except Exception as e:
                        self.log(f"\\n❌ Error: {str(e)}")
                    finally:
                        self.root.after(0, lambda: self.transcribe_btn.config(state='normal'))
            
            # Create and run GUI
            root = tk.Tk()
            app = TranscribeGUI(root, self)
            root.mainloop()
            
        except ImportError:
            print("GUI requires tkinter. Using command line interface instead.")
            self.run_cli()
    
    def run_cli(self, args=None):
        """Run CLI interface"""
        if not args:
            # Interactive mode
            print("Interactive Mode\n")
            
            audio_file = input("Audio file path (drag & drop): ").strip().strip('"')
            if not os.path.exists(audio_file):
                print(f"ERROR: File not found: {audio_file}")
                return
            
            output_name = input("Output name (press Enter for default): ").strip()
            if not output_name:
                output_name = Path(audio_file).stem + "_transcribed"
            
            preserve_fillers = input("Keep filler words? (Y/n): ").strip().lower() != 'n'
            enable_speaker = input("Enable speaker recognition? (Y/n): ").strip().lower() != 'n'
            
            cmd = [self.get_python_exe(), "-m", "transcription.rapid_ultra_processor",
                   audio_file, "-o", output_name]
            
            if not preserve_fillers:
                cmd.append("--no-fillers")
            if not enable_speaker:
                cmd.append("--no-speaker")
                
        else:
            # Direct command mode
            cmd = [self.get_python_exe(), "-m", "transcription.rapid_ultra_processor"] + args
        
        print("\nProcessing...")
        try:
            subprocess.run(cmd)
        except Exception as e:
            print(f"ERROR: {e}")
    
    def diagnose(self):
        """Run diagnostics"""
        print("Running diagnostics...\n")
        
        issues = 0
        
        # Check Python
        print(f"Python version: {sys.version}")
        if sys.version_info >= (3, 8):
            print("✓ Python version OK\n")
        else:
            print("✗ Python 3.8+ required\n")
            issues += 1
        
        # Check virtual environment
        if os.path.exists(self.venv_path):
            print("✓ Virtual environment exists")
        else:
            print("✗ Virtual environment not found")
            issues += 1
        
        # Check packages
        if os.path.exists(self.get_python_exe()):
            print("✓ Python executable found")
            
            # Try importing key packages
            try:
                result = subprocess.run([self.get_python_exe(), "-c", "import whisper"], 
                                      capture_output=True)
                if result.returncode == 0:
                    print("✓ Whisper package installed")
                else:
                    print("✗ Whisper package not installed")
                    issues += 1
            except:
                print("✗ Could not check packages")
                issues += 1
        
        print(f"\nDiagnostic complete. Found {issues} issue(s).")
        
        if issues > 0:
            print("\nRun with --setup to fix issues.")
    
    def main(self):
        """Main entry point"""
        parser = argparse.ArgumentParser(description="Ultra Audio Transcription")
        parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
        parser.add_argument("-o", "--output", help="Output base name")
        parser.add_argument("--gui", action="store_true", help="Launch GUI")
        parser.add_argument("--setup", action="store_true", help="Run setup")
        parser.add_argument("--diagnose", action="store_true", help="Run diagnostics")
        parser.add_argument("--no-fillers", action="store_true", help="Remove filler words")
        parser.add_argument("--no-speaker", action="store_true", help="Disable speaker recognition")
        
        args, unknown = parser.parse_known_args()
        
        self.print_header()
        
        # Handle special commands
        if args.setup:
            self.run_setup()
            return
        
        if args.diagnose:
            self.diagnose()
            return
        
        # Check setup
        if not self.check_setup():
            print("Setup required. Running first-time setup...\n")
            if not self.run_setup():
                return
        
        # Run appropriate interface
        if args.gui or (not args.audio_file and not sys.stdin.isatty()):
            self.run_gui()
        elif args.audio_file:
            # Build command line args
            cli_args = [args.audio_file]
            if args.output:
                cli_args.extend(["-o", args.output])
            if args.no_fillers:
                cli_args.append("--no-fillers")
            if args.no_speaker:
                cli_args.append("--no-speaker")
            cli_args.extend(unknown)  # Pass through any unknown args
            
            self.run_cli(cli_args)
        else:
            # Interactive mode
            self.run_cli()

if __name__ == "__main__":
    app = UltraTranscribe()
    try:
        app.main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        if not IS_EXE:
            import traceback
            traceback.print_exc()
    
    if platform.system() == "Windows" and not IS_EXE:
        input("\nPress Enter to exit...")