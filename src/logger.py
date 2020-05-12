from math import pow
from math import floor
from os import makedirs
from os import get_terminal_size
from os.path import join
from os.path import exists
from os.path import normpath
from os import sep


def conditional_makedir(path):
    try: 
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

    except:
        list_path = normpath(path).split(sep)[1:]
        t = '/'

        for p in list_path:
            t = join(t,p)
            
            if not exists(t):
                makedirs(t)


def retrieve_terminal_size():
    try:
        width = (get_terminal_size().columns - 15)/2
    except OSError:
        width = (80 - 15)/2

    # Just in case
    if (width == None): width = (80 - 15)/2

    return width


def write_summary(filename, model):
    # Open the file
    with open(filename,'w') as f:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def print_summary(f):
    def __print__(s):
        print(s,file=f)
    return __print__


def lr_scheduler(initial_lrate, drop, epochs_drop):
    def step_decay(epoch):
        lrate = initial_lrate * pow(drop, floor((1+epoch)/epochs_drop))
        return lrate
    return step_decay
    
    
