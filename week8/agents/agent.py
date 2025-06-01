import logging

class Agent:
    """
    An abstract superclass for Agents
    Used to log messages in a way that can identify each Agent
    """

    # Foreground colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background color
    BG_BLACK = '\033[40m'
    
    # Reset code to return to default color
    RESET = '\033[0m'

    name: str = ""
    color: str = '\033[37m'

    def log(self, message):
        """
        Log this as an info message, identifying the agent
        """
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)
        
# import logging
# import os
# import datetime

# class Agent:
#     """
#     An abstract superclass for Agents
#     Used to log messages in a way that can identify each Agent
#     """

#     # Foreground colors
#     RED = '\033[31m'
#     GREEN = '\033[32m'
#     YELLOW = '\033[33m'
#     BLUE = '\033[34m'
#     MAGENTA = '\033[35m'
#     CYAN = '\033[36m'
#     WHITE = '\033[37m'
    
#     # Background color
#     BG_BLACK = '\033[40m'
    
#     # Reset code to return to default color
#     RESET = '\033[0m'

#     name: str = ""
#     color: str = '\033[37m'

#     def __init__(self):
#         """
#         Optional initializer to set up log file
#         """
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         log_dir = "logs"
#         os.makedirs(log_dir, exist_ok=True)
#         log_file = f"{log_dir}/{self.__class__.__name__}_{timestamp}.log"
#         self._log_file_path = log_file

#         # Ensure logging is configured once
#         logging.basicConfig(level=logging.INFO)

#     def log(self, message):
#         """
#         Log this as an info message, identifying the agent.
#         Output to console with color, and append to file in plain text.
#         """
#         color_code = self.BG_BLACK + self.color
#         full_message = f"[{self.name}] {message}"
        
#         # Console log (colored)
#         logging.info(color_code + full_message + self.RESET)

#         # Write plain text to file
#         with open(self._log_file_path, "a", encoding="utf-8") as f:
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"[{timestamp}] {full_message}\n")
