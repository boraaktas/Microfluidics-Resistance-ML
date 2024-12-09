import locale
import tkinter as tk

from app import Main_Section


def main():
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'C')
    ROOT = tk.Tk()
    Main_Section(ROOT)
    ROOT.mainloop()


if __name__ == '__main__':
    main()
