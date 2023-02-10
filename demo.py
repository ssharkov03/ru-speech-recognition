"""
Демонстрация использования модуля SpeechRecognitionModule.
"""
import torch
from SpeechRecognitionModule import speech2text

# Setting up parameters
path2audio = "path/to/your/audio"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
verbose = 2  # 0 = just output, 1 = output + time stats, 2 = output + time stats + all in-between outputs

# Getting text from audio
output_text = speech2text(path2audio,
                          device,
                          verbose=2)
print(output_text)
