# TONY/Lexicon/__init__.py
from .LinguisticMarkers import *  

try:
    from .APP import *  
except ImportError:
    pass
except Exception:
    pass  # tkinter può lanciare anche TclError se non c'è display

try:
    from .APP_Colab import *  
except ImportError:
    pass
except Exception:
    pass  
