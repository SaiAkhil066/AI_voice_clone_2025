# 🎤 AI Voice Cloning with XTTS-v2

![Voice Cloning Demo](https://img.shields.io/badge/Voice%20Cloning-XTTS--v2-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

A state-of-the-art voice cloning application using XTTS-v2 technology to recreate natural-sounding speech from just a few seconds of audio.

## ✨ Features

- **High-Quality Voice Cloning**: Uses XTTS-v2 for superior voice reproduction
- **Multi-language Support**: Works with 17 languages including English, Spanish, French, and more
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs for faster processing
- **Voice Similarity Controls**: Fine-tune temperature, repetition penalty, and accent preservation
- **User-friendly Interface**: Clean Gradio web interface with real-time processing
- **Fallback Mode**: Includes SpeechT5 fallback for systems without GPU support

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-voice-cloning.git
   cd ai-voice-cloning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:7860`

## 🖥️ System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- CUDA-compatible GPU (recommended for faster processing)
- Windows, macOS, or Linux

## 📁 Project Structure

```
ai-voice-cloning/
├── improved_voice_clone.py  # Core voice cloning logic
├── app.py                   # Gradio web interface
├── requirements.txt         # Python dependencies
└── README.md
```

## 🎯 Usage Guide

1. **Upload Voice Sample**: Record or upload 3-15 seconds of clear speech
2. **Select Language**: Choose from 17 supported languages
3. **Enter Text**: Type the text you want the voice to speak
4. **Adjust Settings**: Fine-tune voice similarity parameters
5. **Generate**: Click "Generate Speech" and download your result

### Voice Similarity Settings

- **Temperature**: Lower values (0.1-0.7) = more stable but less natural
- **Repetition Penalty**: Higher values (1.5-3.0) = reduces repetitive artifacts
- **Accent Preservation**: Higher values (0.5-0.9) = maintains original accent better

## 🔧 Troubleshooting

### GPU Not Detected

1. Check CUDA availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. Install CUDA-enabled PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify GPU usage in logs:
   ```
   Using GPU: True
   GPU device: [Your GPU Name]
   ```

### Audio Preprocessing Errors

If you encounter audio format issues:
- Ensure audio files are WAV format
- Check sample rate (24kHz recommended)
- Convert stereo to mono if needed

### Memory Issues

For systems with limited GPU memory:
- Reduce batch size in settings
- Use shorter text inputs
- Consider using CPU mode (fallback)

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# Force CPU usage
export USE_FALLBACK=1

# Enable debug logging
export DEBUG=1

# Set custom port
export PORT=8080
```

### Custom Model Loading

Modify `improved_voice_clone.py` to use different models:
```python
self.model_name = "your_custom_model_path"
```

## 📚 API Reference

### ImprovedVoiceCloner Class

```python
cloner = ImprovedVoiceCloner(device="cuda")
cloner.clone_voice(
    voice_sample_path="path/to/sample.wav",
    text="Text to speak",
    output_path="output.wav",
    language="en",
    voice_settings={
        "temperature": 0.65,
        "repetition_penalty": 2.0,
        "accent_strength": 0.8
    }
)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for the XTTS-v2 model
- [Microsoft SpeechT5](https://github.com/microsoft/SpeechT5) for the fallback implementation
- [Gradio](https://gradio.app/) for the web interface

## 🔮 Future Roadmap

- [ ] Batch processing support
- [ ] Real-time voice synthesis
- [ ] Voice style transfer
- [ ] Mobile app integration
- [ ] API endpoint deployment

## 📬 Contact

For questions or support:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: [@yourusername](https://twitter.com/yourusername)

---

<p align="center">Made with ❤️ by Sai Akhil</p>
