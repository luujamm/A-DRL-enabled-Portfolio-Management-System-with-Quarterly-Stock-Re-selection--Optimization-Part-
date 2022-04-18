import sys
from utils.create_test_file import create_testfile
from run import main as run
from utils.render import render


def main():
    args = sys.argv[1:]
    if len(args) > 0:
        create_testfile()
        run()    
    render()


if __name__ == '__main__':
    main()