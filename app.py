# 1. Antisipasi Error audioop pada Python 3.13 di Hugging Face
try:
    import audioop
except ImportError:
    import sys
    import audioop_lts
    sys.modules['audioop'] = audioop_lts

import os

# 2. Menjalankan main.py dengan menyuntikkan argumen 'all' yang diminta program Anda
# Catatan: Jika program Anda membutuhkan kata perintah lain (seperti 'analyze' atau 'preprocess'), 
# silakan ganti kata 'all' di bawah ini sesuai kebutuhan script Anda.
os.system("python src/main.py all")
