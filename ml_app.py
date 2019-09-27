from tkinter import *
from requests import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as svs
import theano as t
import numpy

Height=700
Width=800

def get_weather(city):
    weather_key='982593280c443733b41171d9584114d4'
    url='https://api.openweathermap.org/data/2.5/weather'
    param= {'APPID': weather_key,'q':city,'units':'metric'}
    response= get(url, params=param)
    weather= response.json()

    label['text']=display(weather)
def display(weather):
    try:
        name =weather['name']
        Temperature=weather['main']['temp']
        Pressure=weather['main']['pressure']
        Humidity=weather['main']['humidity']

        #Reading data from csv file
        
        df=pd.read_csv(r"data.csv")

        #Removing Duplicte rows from the dataset

        df.drop_duplicates(inplace=True)

        #Prediction model for deriving Potential Energy from Atmospheric Temperature

        x=t.tensor.fvector('x')
        y=t.tensor.fvector('y')

        #Initializing slope and intercept values

        m=t.shared(-2.16961544,'m')
        c=t.shared(496.98669812,'c')
        yhat=m*x +c     #Line eq
        cost=t.tensor.mean(t.tensor.sqr(y-yhat))/2      #Cost Function
        LR = 0.001      #Learning Rate
        gradm=t.tensor.grad(cost,m)
        gradc=t.tensor.grad(cost,c)
        mn = m - LR*gradm
        cn = c - LR*gradc
        train = t.function(inputs=[x,y],outputs=cost,updates=[(m,mn),(c,cn)])
        df.AT=numpy.array(df.AT).astype('float32')
        df.PE=numpy.array(df.PE).astype('float32')

        #Gradient function
        for i in range(1000):
            cost_val=train(df.AT,df.PE)
        
        #Prediction model for deriving Potential Energy from Exhaust Vacuum
        
        m1=m
        c1=c
        x=t.tensor.fvector('x')
        y=t.tensor.fvector('y')
        m=t.shared(-1.16821583,'m')
        c=t.shared(517.76188736,'c')
        yhat=m*x +c
        cost=t.tensor.mean(t.tensor.sqr(y-yhat))/2
        LR = 0.0001
        gradm=t.tensor.grad(cost,m)
        gradc=t.tensor.grad(cost,c)
        mn = m - LR*gradm
        cn = c - LR*gradc
        train = t.function(inputs=[x,y],outputs=cost,updates=[(m,mn),(c,cn)])
        df.V=numpy.array(df.V).astype('float32')
        df.PE=numpy.array(df.PE).astype('float32')
        for i in range(1000):
            cost_val=train(df.V,df.PE)

        #Prediction model for deriving Potential Energy from Relative Humidity

        m2=m
        c2=c
        x=t.tensor.fvector('x')
        y=t.tensor.fvector('y')
        m=t.shared(0.44338971,'m')
        c=t.shared(420.8747149,'c')
        yhat=m*x +c
        cost=t.tensor.mean(t.tensor.sqr(y-yhat))/2
        LR = 0.000001
        gradm=t.tensor.grad(cost,m)
        gradc=t.tensor.grad(cost,c)
        mn = m - LR*gradm
        cn = c - LR*gradc
        train = t.function(inputs=[x,y],outputs=cost,updates=[(m,mn),(c,cn)])
        df.RH=numpy.array(df.RH).astype('float32')
        df.PE=numpy.array(df.PE).astype('float32')
        for i in range(10000):
            cost_val=train(df.RH,df.PE)
            
        #Prediction model for deriving Potential Energy from Ambient Pressure

        m3=m
        c3=c
        x=t.tensor.fvector('x')
        y=t.tensor.fvector('y')
        m=t.shared(1.3394647136463225,'m')
        c=t.shared(-900.2,'c')
        yhat=m*x +c
        cost=t.tensor.mean(t.tensor.sqr(y-yhat))/2
        LR = 0.000001
        gradm=t.tensor.grad(cost,m)
        gradc=t.tensor.grad(cost,c)
        mn = m - LR*gradm
        cn = c - LR*gradc
        train = t.function(inputs=[x,y],outputs=cost,updates=[(m,mn),(c,cn)])
        df.AP=numpy.array(df.AP).astype('float32')
        df.PE=numpy.array(df.PE).astype('float32')
        for i in range(10000):
            cost_val=train(df.AP,df.PE)

        #Prediction model for deriving Exhaust Vacuum from Ambient Pressure

        m4=m
        c4=c
        x=t.tensor.fvector('x')
        y=t.tensor.fvector('y')
        m=t.shared(1.43776505,'m')
        c=t.shared(26.02951185,'c')
        yhat=m*x +c
        cost=t.tensor.mean(t.tensor.sqr(y-yhat))/2
        LR = 0.0003
        gradm=t.tensor.grad(cost,m)
        gradc=t.tensor.grad(cost,c)
        mn = m - LR*gradm
        cn = c - LR*gradc
        train = t.function(inputs=[x,y],outputs=cost,updates=[(m,mn),(c,cn)])
        df.AT=numpy.array(df.AT).astype('float32')
        df.V=numpy.array(df.V).astype('float32')
        for i in range(1000):
            cost_val=train(df.AT,df.V)

        temp=float(Temperature)
        pres=float(Pressure)
        hum=float(Humidity)
        vol = m.get_value()*temp + c.get_value()

        ans=((m1.get_value() * temp) + (m2.get_value() * vol) + (m4.get_value() * pres) + (m3.get_value() * hum)+(c1.get_value() +c2.get_value() +c4.get_value() + c3.get_value()))/4 

        
        final = 'City: %s \nTemperature: %s℃ \nHumidity: %s%% \nPressure: %s millibar \nPredicted Energy: %s MW\n' %(name,Temperature,Humidity,Pressure,ans)
    except:
        final='Cannot Retrieve Information'
    return final
root = Tk()


canvas = Canvas(root,height=Height, width=Width)
canvas.pack()


bg_img=PhotoImage(file='landscape2.png')
bg_label=Label(root,image=bg_img)
bg_label.place(relwidth=1,relheight=1)


frame =Frame(root,bg="#1d3966",bd=5)
frame.place(relx=0.5,rely=0.35,relheight=0.1,relwidth=0.75,anchor='n')

entry=Entry(frame,font=('Ariel',30,'bold'))
entry.place(relwidth=0.65,relheight=1)

button = Button(frame,text="Submit",font=('Comic Sans MS',20,'bold'),bg='Skyblue',fg='#061733',command= lambda : get_weather(entry.get()))
button.place(relx=0.7,relwidth=0.3,relheight=1)

l_frame=Frame(root,bg="#1d3966",bd=10)
l_frame.place(relx=0.5,rely=0.5,relwidth=0.75,relheight=0.3,anchor='n')

label= Label(l_frame,font=('Ariel',20,'bold'),anchor='nw',justify='left',bg='white')
label.place(relwidth=1,relheight=1)
root.mainloop()
