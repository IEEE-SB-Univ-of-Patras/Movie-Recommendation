import tkinter as tk
from tkinter import PhotoImage
from PIL import ImageTk, Image
import pandas as pd

import NeuralNet

import imdb
import requests
from io import BytesIO
import random

dfMovieData = pd.read_csv("movieOHE.csv")
dfImdbData = pd.read_csv("movieInfo.csv")

listMovieGenres = dfMovieData.columns[3:3 + 19].tolist() #Just removing the "no genres listed"

class MovieRecommenderApp():

    def __init__(self, root):
        
        self.root = root
        root.title("Movie Reccomendation")
        self.canvas = tk.Canvas(root, width=1000, height=1000)
        self.canvas.place(x=0, y=0, anchor=tk.NW)
        
        self.nMoviesPicked = 0
        self.nMaxMoviesPicked = 10 #The total number of movies the user will pick as "Liked" or "Not Liked"
        self.Network = NeuralNet.NeuralNet(dfMovieData)

        self.imDB = imdb.IMDb()

        self.font = "SourceCodePro"
        self.listMoviesReccomend = [ ]
        self.movieSample = None

        self.placeButtons()
        self.askMovie() # Start


    def drawLogo(self):
        
        x, y = 40, 10
        w, h = 600, 100

        x_ = x + 20
        y_ = y + 20

        self.canvas.create_rectangle(x, y, x+w, y+h, outline="#f0a500", fill="#f0a500") 
        self.canvas.create_rectangle(x_, y_, x_+w, y_+h, outline="#e45826", fill="#e45826") 

        self.canvas.create_rectangle(x , y_ + h + 10, x_+1*w, y_+4*h, outline="#e45826", fill="#e45826") 
        
        self.canvas.create_text(70, 50, text = "Movie Reccomendation :)"	, anchor = tk.NW,  font = self.font + " 30")

    def askMovie(self):

        self.movieSample = dfMovieData.sample()
        sTitle = self.movieSample["title"].to_string(index=False)
        
        # GUI Stuff...

        self.drawLogo()

        messagesPrompt = ["Do you like", "How about", "Ever heard of", "Do you think you'd enjoy"]

        self.drawMovie(self.movieSample, 60, 180)

        self.canvas.create_text(50, 150, text = "{} {}?{}".format(random.choice(messagesPrompt), sTitle, " " * 50), anchor = tk.NW,  font = self.font + " 10")
        
            
    def placeButtons(self):
        
        self.buttonY  = tk.Button(text="Yes", font=self.font + " 10", bg="#ef8d32", command=lambda: self.setMoviePreference(True, self.movieSample, True)) 
        self.buttonN  = tk.Button(text="No",font=self.font + " 10", bg="#ef8d32", command=lambda: self.setMoviePreference(True, self.movieSample, False)) 
        self.buttonDN = tk.Button(text="Not Sure", font=self.font + " 10", bg="#ef8d32", command=lambda: self.setMoviePreference(False, None, None)) 
        self.buttonY.place(x=60, y=380) 
        self.buttonN.place(x=140, y=380) 
        self.buttonDN.place(x=220, y=380) 

    def setMoviePreference(self, bPicked, movie, bLiked):
        
        if not bPicked:
            pass
        else:
            self.Network.addMovie(movie, bLiked)
            self.nMoviesPicked += 1
        
        if self.nMoviesPicked < self.nMaxMoviesPicked:
            self.askMovie() 
        else: #Train the Network!

            self.drawLogo()
            self.canvas.create_text(50, 150, text="Our Neural Network is being trained...",anchor = tk.NW,  font = self.font + " 10")

            self.Network.Compile()
            self.Network.Train()
            
            self.recommendMovies()

            self.getBestMovies()
    
    def recommendMovies(self):

        for i in range(100):
            movie = dfMovieData.sample()
            self.listMoviesReccomend.append((movie, self.Network.getPrediction(movie)))

    def getBestMovies(self):

        self.buttonY.destroy()
        self.buttonN.destroy()
        self.buttonDN.destroy()
        self.buttonDN.destroy()
        self.buttonDN.destroy()

        self.nCurrentMovie = 0
        
        self.getNextMovie()


        self.listMoviesReccomend.sort(key = lambda movie : movie[1], reverse=True) # Sort by highest rated movies. 
        
        buttonNextMovie = tk.Button(text="Next Reccomendation", font=self.font + " 10", command = self.getNextMovie)
        buttonNextMovie.place(x = 60,y=380)

    
    def getNextMovie(self):
        
        self.drawLogo()
        
        self.nCurrentMovie  = (self.nCurrentMovie + 1) % 10
        movie = self.listMoviesReccomend[self.nCurrentMovie][0]
   
        messagesPrompt = ["We think you'd like", "You might enjoy", "I'm sure you'll like", "You'll probably like"]

        self.canvas.create_text(50, 150, text = "{} {}{}".format(random.choice(messagesPrompt), movie["title"].to_string(index=False), " " * 50), 
                anchor = tk.NW,  font = self.font + " 10")
        
        self.drawMovie(movie, 60, 180)

    def drawMovie(self, movie, x_, y_):


        rowImdb = dfImdbData[dfImdbData["movieId"] == int(movie["movieId"])] #It works!
        movieIMDB = self.imDB.get_movie(rowImdb["imdbId"])
        
        imgUrl = movieIMDB["cover url"]
        
        imgResponse = requests.get(imgUrl)

        imgMovie = Image.open( BytesIO( imgResponse.content ) )
        
        #self.tkImage = ImageTk.PhotoImage( Image.open( imgMovie ).resize((200, 300)), Image.ANTIALIAS ) #Image resizing (doesn't work for now.)
        self.tkImage = ImageTk.PhotoImage( imgMovie )
        self.canvas.create_image(x_, y_, image = self.tkImage, anchor=tk.NW) 
        
        for i, genre in enumerate(movie["genres"].to_string(index=False).split("|")):

            self.canvas.create_text(440, 180 + 20 * i, text="* {}".format(genre), anchor=tk.NW, font=self.font + " 10")
def main():

    root = tk.Tk()
    root.geometry("700x500")
    myapp = MovieRecommenderApp(root)
    root.mainloop()

main()
