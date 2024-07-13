# run this on the raspberry pi

import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO
import socket

def get_ip_address():
    try:
        # Attempts to create a socket connection to determine the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google's DNS server
        ip = str(s.getsockname()[0])
        s.close()
    except:
        ip = "localhost"  # Fallback to localhost if unable to determine IP
    return ip

# Setup GPIO
GPIO.setmode(GPIO.BCM) # Broadcom pin numbering scheme
GPIO.setup(18, GPIO.OUT) # 18 pin set as output

def on_connect(client, userdata, flag, rc):
    client.subscribe("test")
    
def on_message(client, userdata, msg):
    message = msg.payload.decode()
    print(f"Message received: {message}")
    if message == "1":
        GPIO.output(18, GPIO.HIGH)
        print("GPIO 18 turned on")
    else:
        GPIO.output(18, GPIO.LOW)
        print("GPIO 18 turned off")

# broker_address = "localhost"
broker_address = get_ip_address()
client = mqtt.Client("Subscriber")
client.on_message = on_message
client.on_connect = on_connect
client.connect(broker_address, 1883, 60)
client.loop_forever()
