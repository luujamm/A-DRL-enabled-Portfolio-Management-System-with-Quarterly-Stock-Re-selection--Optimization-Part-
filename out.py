import sys

from run import main as run
from src.utils.create_test_file import create_testfile
from src.utils.render import render


def main():
    args = sys.argv[1:]
    if len(args) > 0:
        create_testfile()
        run()    
    render()


if __name__ == '__main__':
    main()