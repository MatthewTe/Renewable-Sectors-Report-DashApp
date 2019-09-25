# Importing base finance models:
from Base_Finance_models import Security, ETF

# Creating Natural Gas pricing model:
class Natural_Gas(ETF):
    """docstring for Natural_Gas."""

    def __init__(self):

        # Initilazing parent ETF() object:
        super().__init__(parent_ticker) # Decide price ETF.
