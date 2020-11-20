from tkinter import *

import Search1

root = Tk()
query1 = StringVar()
root.minsize(400,400)
def field():
    value1 = query1.get()
    Search1.search(value1)

title = Label(root, text="Potential research supervisor").grid(row=6, column=5)
query = Entry(root, textvariable = query1).grid(row=8, column=5)

button1 = Button(root, text=" Enter text for Search", fg="blue",
                 command=field).grid(row=10, column=5)

root.mainloop()
