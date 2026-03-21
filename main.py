import kivy
from kivy.app import App
from kivy.uix.label import Label
import shtns
import numpy as np

class MyApp(App):
    def build(self):
        return Label(text='Hello SHTNS')

if __name__ == '__main__':
    MyApp().run()
