from guizero import App, Slider, TextBox


def slider_changed1(slider_value):
    textbox1.value = slider_value

def slider_changed2(slider_value):
    textbox2.value = slider_value


app = App(title="Reinforcement Learning")
slider1 = Slider(app, command=slider_changed1)
textbox1 = TextBox(app)

slider2 = Slider(app, command=slider_changed2)
textbox2 = TextBox(app)

app.display()
