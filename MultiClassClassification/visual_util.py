CONSOLE_COLOR_CODES = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'purple': '\033[95m',
    'cyan': '\033[96m',
}

END_COLOR_CODE = '\033[0m'


class ColoredPrint():
    
    def __init__(self):
        pass

    @staticmethod
    def red(text: str):
        ColoredPrint.cprint(text, 'red')
    
    @staticmethod     
    def green(text: str):
        ColoredPrint.cprint(text, 'green')
        
    @staticmethod
    def yellow(text: str):
        ColoredPrint.cprint(text, 'yellow')
    
    @staticmethod
    def blue(text: str):
        ColoredPrint.cprint(text, 'blue')
    
    @staticmethod
    def purple(text: str):
        ColoredPrint.cprint(text, 'purple')
    
    @staticmethod
    def cyan(text: str):
        ColoredPrint.cprint(text, 'cyan')

    @staticmethod
    def cprint(text: str, color: str):
        print(f"{CONSOLE_COLOR_CODES.get(color, '')}{text}{END_COLOR_CODE}")
        

if __name__ == '__main__':
    
    ColoredPrint.red("Prova")
    ColoredPrint.green("Prova")
    ColoredPrint.yellow("Prova")
    ColoredPrint.blue("Prova")
    ColoredPrint.purple("Prova")
    ColoredPrint.cyan("Prova")