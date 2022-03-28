from pyfirmata import Arduino,SERVO #pifirmata is serial communication b/w python and arduino
import time

port = 'COM4'
pin = 10
board = Arduino(port)
led = board.get_pin('d:12:o') #getting led pin "0"-off '1'-on

board.digital[pin].mode = SERVO
def servo_rotate(pin,angle): #function to acces servo
    board.digital[pin].write(angle)

def led_motor(y):
    if y==1:
        servo_rotate(pin,90)
        led.write(1)

        time.sleep(5) #delay of 5sec

        for i in range(90,0,-1):
            servo_rotate(pin,i)
        led.write(0)    
          

    