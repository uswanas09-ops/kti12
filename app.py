try:
    import audioop
except ImportError:
    import sys
    import audioop_lts
    sys.modules['audioop'] = audioop_lts

import os
os.system("python src/main.py")
