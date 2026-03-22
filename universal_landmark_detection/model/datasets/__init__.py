from .cephalometric import Cephalometric
from .hand import Hand
from .chest import Chest
from .hip import Hip

def get_dataset(s):
    return {
            'cephalometric':Cephalometric,
            'hand':Hand,
            'chest':Chest,
            'hip':Hip,
           }[s.lower()]
