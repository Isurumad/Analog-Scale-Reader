from tkinter import*

def display(image):
    window = Tk()
    window.title('Digital Meter')
    canvas = Canvas(window,width=500,height=500)
    canvas,pack()
    chras.create_image(0,0,anchor=NW,image=image)
