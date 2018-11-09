# print(v_)
import locale
import tkinter as tk
from tkinter import *
from modelrestore import vec_sim


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
            self.digits = 5

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

def sel():
   selection1 = "Value = " + str(var1.get()*0.000001)
   selection2 = "Value = " + str(var2.get()*0.000001)
   vec.append(var1.get()*0.000001)
   vec.append(var2.get()*0.000001)

   label1.config(text = selection1)
   label2.config(text = selection2)

   output = vec_sim(vec,5)
   for i in range (len(output)):
       mylist.insert(END,str(output[i]))


if __name__ == '__main__':
    master = tk.Tk()
    master.geometry("500x500")
    #display_text = tk.StringVar()
    #display = tk.Label(master, textvariable=display_text)
    var1 = DoubleVar()
    var2 = DoubleVar()
    frame = Frame(master)
    frame.pack()
    vec = []
    bottomframe = Frame(master)
    bottomframe.pack(side=LEFT)

    topframe = Frame(master)
    topframe.pack(side=LEFT)

    scrollbar = Scrollbar(master)
    scrollbar.pack(side=RIGHT, fill=Y,expand=True)
    mylist = Text(master,wrap=WORD, yscrollcommand=scrollbar.set)
    mylist.pack(side=LEFT, fill=BOTH ,expand=True)
    scrollbar.config(command=mylist.yview)

    w1 = NewScale(topframe, from_=-2, to=1, resolution=0.000001,variable = var1 )
    w2 = NewScale(bottomframe, from_=-2, to=1, resolution=0.000001,variable = var2 )
    #T = Text(topframe, height=2, width=30)
    button1 = Button(topframe, text="Get Scale Value", command=sel)
    button2 = Button(bottomframe, text="Get Scale Value", command=sel)
    label1 = Label(topframe)
    label2 = Label(bottomframe)

    w1.pack(anchor=CENTER, expand=True)
    button1.pack(anchor=CENTER)
    label1.pack(anchor=CENTER)

    w2.pack(anchor=CENTER, expand=True)
    button2.pack(anchor=CENTER)
    label2.pack(anchor=CENTER)


    #T.pack(side=LEFT, padx=10)
    #display.pack()

    master.mainloop()

