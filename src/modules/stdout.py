from contextlib import redirect_stdout
from os.path import join

class Stdout_parse():
    def __init__(self,filename="output.txt",folder="", writing_type = "w+", print_to_terminal = False):
        self.folder = folder
        self.filename = filename
        self.f_obj = None
        self.writing_type = writing_type
        self.print_term = print_to_terminal

    def __enter__(self):
        self.f_obj = open(join(self.folder, self.filename), self.writing_type)
        self.redirect = redirect_stdout(self.f_obj)
        self.redirect.__enter__()
        return self
                  
    def __exit__(self,exc_type, exc_value, traceback):
        self.redirect.__exit__(exc_type, exc_value, traceback)
        self.f_obj.close()
        if self.print_term:
            with open(join(self.folder, self.filename), 'r') as f:
                for line in f.readlines():
                    print(line, end="")

            
        return True