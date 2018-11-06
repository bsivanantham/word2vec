# print(v_)
import locale
import tkinter as tk
from tkinter import *


class NewScale(tk.Frame):
    def __init__(self, master=None, **options):
        tk.Frame.__init__(self, master)

        # Disable normal value display...
        options['showvalue'] = False
        # ... and use custom display instead
        options['command'] = self._on_scale

        # Set resolution to 1 and adjust to & from value
        self.res = options.get('resolution', 1)
        from_ = int(options.get('from_', 0) / self.res)
        to = int(options.get('to', 100) / self.res)
        options.update({'resolution': 1, 'to': to, 'from_': from_})

        # This could be improved...
        if 'digits' in options:
            self.digits = ['digits']
            del options['digits']
        else:
            self.digits = 2

        self.scale = tk.Scale(self, **options)
        self.scale_label = tk.Label(self)
        orient = options.get('orient', tk.VERTICAL)
        if orient == tk.VERTICAL:
            side, fill = 'right', 'y'
        else:
            side, fill = 'top', 'x'
        self.scale.pack(side=side, fill=fill)
        self.scale_label.pack(side=side)

    def _on_scale(self, value):
        value = locale.atof(value) * self.res
        value = locale.format_string('%.*f', (self.digits, value))
        self.scale_label.configure(text=value)


if __name__ == '__main__':
    master = tk.Tk()
    master.geometry("500x500")
    display_text = tk.StringVar()
    display = tk.Label(master, textvariable=display_text)

    w1 = NewScale(master, from_=-2, to=1, resolution=0.000001)
    w2 = NewScale(master, from_=-2, to=1, resolution=0.000001)
    T = Text(master, height=2, width=30)
    w1.pack(side=LEFT, expand=True)
    w2.pack(side=LEFT, expand=True)
    T.pack(side=LEFT, padx=10)
    display.pack()

    display_text.set("Output vector")
    master.mainloop()
